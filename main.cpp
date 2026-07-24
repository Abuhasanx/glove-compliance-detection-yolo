/*
 * ============================================================
 *  Gloved vs Ungloved Hand Detection — C++ Inference Script
 *  Author : Abu Hasan
 *  Mail   : abuhasanxs@gmail.com
 *
 *  Part 1: Gloved vs Ungloved Hand Detection (Practical Task)
 *  Scenario: Safety compliance system that checks whether
 *  workers are wearing gloves. Can be deployed on video
 *  streams or snapshots from factory cameras.
 *
 *  Dataset: https://www.kaggle.com/datasets/innominate817/hagrid-classification-512p
 *  Dataset: https://www.kaggle.com/datasets/yashdev01/gloves-and-bare-hands-datasets
 *
 *  NOTE: Work done under 24 hrs with limited resources.
 *        Model: YOLOv8s exported to ONNX
 *
 *  Build:
 *    g++ main.cpp -o glove_detect \
 *        $(pkg-config --cflags --libs opencv4) \
 *        -std=c++17
 *
 *  OR with CMake — see README.
 * ============================================================
 */

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <iomanip>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

namespace fs = std::filesystem;

// ============================================================
//  CONFIG — edit these paths before running
// ============================================================

// Model path — only ONNX supported in this C++ build
static const std::string MODEL_PATH   = "D:/vision/GLOVE TEST FILES/best.onnx";

static const std::string INPUT_FOLDER  = "D:/vision/images to test/input";
static const std::string OUTPUT_FOLDER = "D:/vision/images to test/output";

// Global low threshold — lets weak detections reach the per-class filter
static const float CONF_THRESH = 0.1f;
static const float IOU_THRESH  = 0.5f;
static const int   IMGSZ       = 640;

// Per-class confidence floors (key = lowercase class name from model)
static const std::unordered_map<std::string, float> PER_CLASS_CONF = {
    {"gloves", 0.30f},
    {"glove",  0.30f},
    {"hand",   0.15f},
};

// Supported image extensions
static const std::vector<std::string> IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
};

// ============================================================
//  DISPLAY STYLING
// ============================================================

static const int BOX_THICKNESS  = 2;
static const double FONT_SCALE  = 0.5;
static const int FONT_THICKNESS = 1;
static const int FONT_FACE      = cv2::FONT_HERSHEY_SIMPLEX;

// BGR colours
static const cv::Scalar COLOR_GLOVE    (255,   0,   0);   // Blue  → GLOVED
static const cv::Scalar COLOR_HAND     (  0,   0, 255);   // Red   → UN GLOVED
static const cv::Scalar COLOR_DEFAULT  (  0, 255,   0);   // Green → unknown
static const cv::Scalar COLOR_LABEL_BG (  0,   0,   0);   // Black label bg
static const cv::Scalar COLOR_LABEL_TXT(255, 255, 255);   // White label text

struct LabelStyle {
    std::string text;
    cv::Scalar  color;
};

static const std::unordered_map<std::string, LabelStyle> LABEL_STYLE = {
    {"gloves", {"GLOVED",     COLOR_GLOVE}},
    {"glove",  {"GLOVED",     COLOR_GLOVE}},
    {"hand",   {"UN GLOVED",  COLOR_HAND }},
};

// ============================================================
//  HELPER UTILITIES
// ============================================================

// Convert string to lowercase
std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

// Return true if the file extension is in our supported list
bool isSupportedImage(const fs::path& p) {
    std::string ext = toLower(p.extension().string());
    for (const auto& e : IMAGE_EXTS)
        if (ext == e) return true;
    return false;
}

// Look up label style for a raw class name
LabelStyle styleFor(const std::string& rawName) {
    std::string key = toLower(rawName);
    auto it = LABEL_STYLE.find(key);
    if (it != LABEL_STYLE.end()) return it->second;
    // fallback: uppercase name, green box
    std::string upper = rawName;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    return {upper, COLOR_DEFAULT};
}

// Draw bounding box + label on image
void drawDetection(cv::Mat& img,
                   int x1, int y1, int x2, int y2,
                   const std::string& labelText,
                   const cv::Scalar& color)
{
    // Bounding box
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                  color, BOX_THICKNESS, cv::LINE_8);

    // Measure text
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(labelText, FONT_FACE,
                                        FONT_SCALE, FONT_THICKNESS, &baseline);
    int pad = 4;

    // Label background — above box if it fits, inside top otherwise
    int bgTop, bgBottom;
    if (y1 - textSize.height - baseline - pad < 0) {
        bgTop    = y1;
        bgBottom = y1 + textSize.height + baseline + pad;
    } else {
        bgTop    = y1 - textSize.height - baseline - pad;
        bgBottom = y1;
    }

    cv::rectangle(img,
                  cv::Point(x1, bgTop),
                  cv::Point(x1 + textSize.width + 2 * pad, bgBottom),
                  COLOR_LABEL_BG, cv::FILLED);

    int textY = bgBottom - baseline - 2;
    cv::putText(img, labelText,
                cv::Point(x1 + pad, textY),
                FONT_FACE, FONT_SCALE,
                COLOR_LABEL_TXT, FONT_THICKNESS, cv::LINE_AA);
}

// ============================================================
//  YOLO ONNX POST-PROCESSING
//
//  YOLOv8 ONNX output tensor shape: [1, num_classes+4, num_anchors]
//  Layout per anchor: [cx, cy, w, h, cls0_score, cls1_score, ...]
// ============================================================

struct Detection {
    std::string label;
    float       confidence;
    cv::Rect    box;
};

std::vector<Detection> postprocess(
    const cv::Mat&              output,      // raw model output [1, C+4, N]
    const std::vector<std::string>& classNames,
    int                         origW,
    int                         origH,
    float                       confThresh,
    float                       iouThresh)
{
    // Reshape to [num_classes+4, num_anchors]
    // output.size[0]=1, output.size[1]=C+4, output.size[2]=N
    int rows    = output.size[1];   // 4 + num_classes
    int anchors = output.size[2];
    int numClasses = rows - 4;

    // Scale factors from 640x640 back to original image size
    float scaleX = static_cast<float>(origW) / IMGSZ;
    float scaleY = static_cast<float>(origH) / IMGSZ;

    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    std::vector<int>       classIds;

    const float* data = reinterpret_cast<const float*>(output.data);

    for (int a = 0; a < anchors; ++a) {
        // cx, cy, w, h are in rows 0-3
        float cx = data[0 * anchors + a];
        float cy = data[1 * anchors + a];
        float w  = data[2 * anchors + a];
        float h  = data[3 * anchors + a];

        // Find the best class score
        int   bestCls   = -1;
        float bestScore = 0.0f;
        for (int c = 0; c < numClasses; ++c) {
            float score = data[(4 + c) * anchors + a];
            if (score > bestScore) {
                bestScore = score;
                bestCls   = c;
            }
        }

        if (bestScore < confThresh || bestCls < 0) continue;

        // xyxy in 640-space → scale to original
        int x1 = static_cast<int>((cx - w / 2.0f) * scaleX);
        int y1 = static_cast<int>((cy - h / 2.0f) * scaleY);
        int bw = static_cast<int>(w * scaleX);
        int bh = static_cast<int>(h * scaleY);

        boxes   .push_back(cv::Rect(x1, y1, bw, bh));
        scores  .push_back(bestScore);
        classIds.push_back(bestCls);
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confThresh, iouThresh, indices);

    std::vector<Detection> detections;
    for (int idx : indices) {
        std::string rawName = (bestCls < (int)classNames.size())
                                  ? classNames[classIds[idx]]
                                  : "unknown";

        // Per-class confidence floor
        std::string key = toLower(rawName);
        float floor = confThresh;
        auto it = PER_CLASS_CONF.find(key);
        if (it != PER_CLASS_CONF.end()) floor = it->second;

        if (scores[idx] < floor) continue;

        LabelStyle style = styleFor(rawName);
        detections.push_back({style.text, scores[idx], boxes[idx]});
    }

    return detections;
}

// ============================================================
//  SIMPLE JSON WRITER (no external library needed)
// ============================================================

std::string escapeJson(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        if (c == '"')       o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else                o << c;
    }
    return o.str();
}

void writeJson(const fs::path& jsonPath,
               const std::string& filename,
               const std::vector<Detection>& dets)
{
    std::ofstream f(jsonPath);
    f << "{\n";
    f << "  \"filename\": \"" << escapeJson(filename) << "\",\n";
    f << "  \"detections\": [\n";
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        f << "    {\n";
        f << "      \"label\": \""      << escapeJson(d.label) << "\",\n";
        f << std::fixed << std::setprecision(4);
        f << "      \"confidence\": "   << d.confidence << ",\n";
        f << "      \"bbox\": ["
          << d.box.x << ", "
          << d.box.y << ", "
          << (d.box.x + d.box.width)  << ", "
          << (d.box.y + d.box.height) << "]\n";
        f << "    }";
        if (i + 1 < dets.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n}\n";
}

// ============================================================
//  MAIN
// ============================================================

int main()
{
    // --- Validate paths ---
    if (!fs::exists(MODEL_PATH)) {
        std::cerr << "ERROR: model not found at " << MODEL_PATH << "\n";
        return 1;
    }

    fs::path inputDir(INPUT_FOLDER);
    fs::path outputDir(OUTPUT_FOLDER);

    if (!fs::exists(inputDir)) {
        std::cerr << "ERROR: input folder not found: " << inputDir << "\n";
        return 1;
    }
    fs::create_directories(outputDir);

    // --- Collect image paths ---
    std::vector<fs::path> imagePaths;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && isSupportedImage(entry.path()))
            imagePaths.push_back(entry.path());
    }
    std::sort(imagePaths.begin(), imagePaths.end());

    if (imagePaths.empty()) {
        std::cerr << "ERROR: no supported images found in " << inputDir << "\n";
        return 1;
    }
    std::cout << "Found " << imagePaths.size() << " images in " << inputDir << "\n";

    // --- Load ONNX model via OpenCV DNN ---
    std::cout << "Loading ONNX model: " << MODEL_PATH << "\n";
    cv::dnn::Net net = cv::dnn::readNetFromONNX(MODEL_PATH);

    // Use GPU (CUDA) if available, otherwise CPU
#ifdef USE_CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::cout << "Backend: CUDA\n";
#else
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Backend: CPU (OpenCV)\n";
#endif

    // Class names — must match your training order
    // Edit this list to match your model's class names exactly
    std::vector<std::string> classNames = {"gloves", "hand"};

    int totalDetections          = 0;
    int totalDroppedByClassConf  = 0;

    // --- Inference loop ---
    for (size_t idx = 0; idx < imagePaths.size(); ++idx) {
        const auto& imgPath = imagePaths[idx];

        cv::Mat img = cv::imread(imgPath.string());
        if (img.empty()) {
            std::cerr << "  WARNING: could not read " << imgPath.filename() << ", skipping.\n";
            continue;
        }

        int origH = img.rows;
        int origW = img.cols;

        // Pre-process: resize to IMGSZ x IMGSZ, normalize to [0,1]
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob,
                               1.0 / 255.0,
                               cv::Size(IMGSZ, IMGSZ),
                               cv::Scalar(),
                               true,    // swapRB: BGR → RGB
                               false);  // crop

        net.setInput(blob);

        // Forward pass
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Post-process
        // YOLOv8 produces one output tensor; reshape if needed
        cv::Mat rawOut = outputs[0];
        // Ensure shape is [1, rows, anchors] — some exports are [1, anchors, rows]
        // YOLOv8 standard export: [1, 4+classes, 8400]
        if (rawOut.dims == 3 && rawOut.size[1] < rawOut.size[2]) {
            // already in correct layout
        } else if (rawOut.dims == 3 && rawOut.size[1] > rawOut.size[2]) {
            // transpose [1, 8400, 84] → [1, 84, 8400]
            cv::Mat transposed;
            // flatten to 2D, transpose, then reshape
            cv::Mat mat2d(rawOut.size[1], rawOut.size[2], CV_32F, rawOut.data);
            cv::Mat t2d = mat2d.t();
            int newSizes[3] = {1, t2d.cols, t2d.rows};
            rawOut = t2d.reshape(1, 3, newSizes).clone();
        }

        // Count dropped boxes separately
        int beforeNMS = 0;
        {
            // Quick pre-count before per-class filter (approximate)
            int rows_    = rawOut.size[1];
            int anchors_ = rawOut.size[2];
            int nCls_    = rows_ - 4;
            const float* d = reinterpret_cast<const float*>(rawOut.data);
            for (int a = 0; a < anchors_; ++a) {
                float best = 0;
                for (int c = 0; c < nCls_; ++c)
                    best = std::max(best, d[(4 + c) * anchors_ + a]);
                if (best >= CONF_THRESH) ++beforeNMS;
            }
        }

        std::vector<Detection> detections = postprocess(
            rawOut, classNames, origW, origH, CONF_THRESH, IOU_THRESH);

        totalDroppedByClassConf += (beforeNMS - static_cast<int>(detections.size()));

        // Draw detections
        for (const auto& det : detections) {
            LabelStyle style = styleFor(det.label);  // re-lookup for color
            // find color from LABEL_STYLE by matching label text
            cv::Scalar color = COLOR_DEFAULT;
            for (const auto& kv : LABEL_STYLE) {
                if (kv.second.text == det.label) { color = kv.second.color; break; }
            }

            int x1 = det.box.x;
            int y1 = det.box.y;
            int x2 = det.box.x + det.box.width;
            int y2 = det.box.y + det.box.height;

            // Clamp to image bounds
            x1 = std::max(0, x1); y1 = std::max(0, y1);
            x2 = std::min(origW - 1, x2); y2 = std::min(origH - 1, y2);

            drawDetection(img, x1, y1, x2, y2, det.label, color);
        }

        totalDetections += static_cast<int>(detections.size());

        // Save annotated image
        fs::path outImgPath = outputDir / imgPath.filename();
        cv::imwrite(outImgPath.string(), img);

        // Save JSON log
        fs::path jsonPath = outputDir / (imgPath.stem().string() + ".json");
        writeJson(jsonPath, imgPath.filename().string(), detections);

        std::cout << "[" << (idx + 1) << "/" << imagePaths.size() << "] "
                  << imgPath.filename().string()
                  << ": " << detections.size() << " detection(s)\n";
    }

    std::cout << "\nDone. " << totalDetections
              << " total detections across " << imagePaths.size() << " images.\n";
    std::cout << "(" << totalDroppedByClassConf
              << " candidate boxes were filtered out by PER_CLASS_CONF)\n";
    std::cout << "Annotated images + JSON logs saved to: " << outputDir << "\n";

    return 0;
}
