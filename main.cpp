// 第二课：选题B 学生证识别 —— C++ 完整版（颜色线索 + 边缘线索）
// ============================================================
// 对应 Python 版 student_card_detect.py 的核心链路：
//   读图 → 缩放到处理分辨率
//        ├─ 颜色线索：中值滤波 → HSV 阈值 → 形态学 → 找轮廓
//        └─ 边缘线索：中值滤波 → CLAHE → 灰度 Canny + Lab 色度梯度 → 闭运算 → 找轮廓
//   两条线索各自的候选 → 几何筛选 → 打分取最优 → 四点排序 → 透视校正
//
// 为什么必须有两条线索（实测结论）：
//   - 举着卡、深色背景：颜色线索最准（卡片色相明显）
//   - 卡平放在桌面上（手机照片）：边缘线索更稳（整卡轮廓清晰）
//   只实现一条都会漏检。
//
// 用法：
//   NO_WINDOW=1 ./lesson2_card 图片路径 [输出前缀]   # 无窗口运行（脚本/远程用）
//   ./lesson2_card 图片路径 [输出前缀]               # 弹窗显示
//   ./lesson2_card                                   # 摄像头第一帧
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdlib>

// ================= 参数区（与 Python 版一一对应） =================
const int    PROC_W      = 640;      // 摄像头处理宽度
const int    PROC_W_IMG  = 800;      // 离线图片处理宽度（实测 640~800 是稳定区间，再高会失效）
const int    OUT_W       = 428;      // 证件 85.6:54 ≈ 1.585
const int    OUT_H       = 270;
const int    MEDIAN_K    = 5;

// 颜色线索
const double MIN_AREA_C  = 0.02;
const double RECT_MIN_C  = 0.60;
const int    COLOR_CLOSE = 21;       // 闭运算核（填掉卡面图案空洞）

// 边缘线索
const double MIN_AREA_E  = 0.05;
const double RECT_MIN_E  = 0.82;     // 真卡 0.86~0.96，褶皱杂物 0.59~0.72
const double BORDER_MIN  = 22.0;
const double BORDER_RATI = 1.25;

// 共用
const double MAX_AREA    = 0.62;
const double AR_LO       = 1.10;
const double AR_HI       = 2.20;

// 青蓝预设（按这张校园卡实测标定）：H 85~120, S≥30, V≥45
const cv::Scalar LO(85, 30, 45);
const cv::Scalar HI(120, 255, 255);

// ================= 工具函数 =================
std::vector<cv::Point2f> orderPoints(std::vector<cv::Point2f> pts) {
    cv::Point2f c(0, 0);
    for (const auto& p : pts) c += p;
    c *= (1.0f / (float)pts.size());
    std::sort(pts.begin(), pts.end(), [&](const cv::Point2f& a, const cv::Point2f& b) {
        return std::atan2(a.y - c.y, a.x - c.x) < std::atan2(b.y - c.y, b.x - c.x);
    });
    int k = 0; float best = 1e9f;
    for (int i = 0; i < 4; ++i) {
        float s = pts[i].x + pts[i].y;
        if (s < best) { best = s; k = i; }
    }
    std::rotate(pts.begin(), pts.begin() + k, pts.end());
    cv::Point2f v1 = pts[1] - pts[0], v2 = pts[2] - pts[0];
    if (v1.x * v2.y - v1.y * v2.x < 0) std::swap(pts[1], pts[3]);
    return pts;
}

double aspectRatio(const std::vector<cv::Point2f>& q) {
    double w = (cv::norm(q[1] - q[0]) + cv::norm(q[2] - q[3])) * 0.5;
    double h = (cv::norm(q[3] - q[0]) + cv::norm(q[2] - q[1])) * 0.5;
    if (h < 1e-6) return 0.0;
    double ar = w / h;
    return ar >= 1.0 ? ar : 1.0 / ar;
}

double arPrior(double ar, double target = 85.6 / 54.0) {
    double d = (ar - target) / 0.8;
    return std::exp(-0.5 * d * d);
}

// 沿四边形四条边采样梯度，衡量"边是不是真的压在边缘上"
void borderSupport(const cv::Mat& grad, const std::vector<cv::Point2f>& q,
                   double& border, double& ratio) {
    border = 0; ratio = 0;
    int n = 0; double sum = 0;
    for (int i = 0; i < 4; ++i) {
        cv::Point2f p1 = q[i], p2 = q[(i + 1) % 4];
        for (int t = 0; t < 48; ++t) {
            float u = 0.04f + (1.0f - 0.08f) * t / 47.0f;
            int x = cvRound(p1.x + u * (p2.x - p1.x));
            int y = cvRound(p1.y + u * (p2.y - p1.y));
            int x0 = std::max(0, x - 2), y0 = std::max(0, y - 2);
            int x1 = std::min(grad.cols - 1, x + 2), y1 = std::min(grad.rows - 1, y + 2);
            if (x1 < x0 || y1 < y0) continue;
            double mx = 0;
            cv::minMaxLoc(grad(cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)), nullptr, &mx);
            sum += mx; ++n;
        }
    }
    if (n == 0) return;
    border = sum / n;

    cv::Mat m = cv::Mat::zeros(grad.size(), CV_8U);
    std::vector<cv::Point> qi;
    for (const auto& p : q) qi.emplace_back(cvRound(p.x), cvRound(p.y));
    cv::fillPoly(m, std::vector<std::vector<cv::Point>>{qi}, 255);
    cv::erode(m, m, cv::getStructuringElement(cv::MORPH_RECT, {11, 11}));
    if (cv::countNonZero(m) < 50) return;
    double interior = cv::mean(grad, m)[0];
    ratio = border / std::max(interior, 1.0);
}

struct Candidate {
    std::vector<cv::Point2f> quad;
    double score = -1, areaPct = 0, ar = 0, rect = 0, border = 0;
    std::string source;
};

void collect(const cv::Mat& binary, const cv::Mat& grad, double frameArea,
             bool fromColor, std::vector<Candidate>& out) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    if (std::getenv("DEBUG_CAND")) {
        std::cout << "  [" << (fromColor ? "颜色" : "边缘") << "] 轮廓 "
                  << contours.size() << " 个，前几个候选：" << std::endl;
        int shown = 0;
        for (const auto& c : contours) {
            double a = cv::contourArea(c);
            if (a < 0.01 * frameArea) continue;
            cv::RotatedRect r2 = cv::minAreaRect(c);
            double rr2 = a / std::max((double)r2.size.width * r2.size.height, 1e-6);
            std::cout << "      面积=" << a / frameArea * 100 << "% 矩形度=" << rr2
                      << " 外接框=" << r2.size.width << "x" << r2.size.height << std::endl;
            if (++shown >= 5) break;
        }
    }

    for (const auto& cnt : contours) {
        double area = cv::contourArea(cnt);
        double minArea = fromColor ? MIN_AREA_C : MIN_AREA_E;
        if (area < minArea * frameArea || area > MAX_AREA * frameArea) continue;

        cv::RotatedRect rr = cv::minAreaRect(cnt);
        double rect = area / std::max((double)rr.size.width * rr.size.height, 1e-6);
        if (rect < (fromColor ? RECT_MIN_C : RECT_MIN_E)) continue;

        cv::Point2f box[4];
        rr.points(box);
        std::vector<cv::Point2f> quad = orderPoints({box[0], box[1], box[2], box[3]});

        double ar = aspectRatio(quad);
        if (ar < AR_LO || ar > AR_HI) continue;

        double border = 0, ratio = 0;
        if (!fromColor) {
            borderSupport(grad, quad, border, ratio);
            if (border < BORDER_MIN || ratio < BORDER_RATI) continue;
        }

        Candidate c;
        c.quad = quad;
        c.areaPct = area / frameArea * 100.0;
        c.ar = ar;
        c.rect = rect;
        c.border = border;
        c.source = fromColor ? "颜色" : "边缘";
        c.score = arPrior(ar) * rect * (area / frameArea) *
                  (fromColor ? 1.0 : std::min(border, 150.0) / 150.0);
        out.push_back(c);
    }
}

int main(int argc, char** argv) {
    std::cout << "OpenCV " << CV_VERSION << " —— 学生卡识别 + 透视校正" << std::endl;

    const bool useCamera = (argc <= 1);          // 不带参数 = 实时摄像头；带参数 = 单张图片
    cv::VideoCapture cap;
    if (useCamera) {
        cap.open(0, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        if (!cap.isOpened()) { std::cerr << "打不开摄像头\n"; return 1; }
        cv::namedWindow("Student Card Detect", cv::WINDOW_NORMAL);
        cv::resizeWindow("Student Card Detect", 1000, 420);
    }

    const std::vector<cv::Point2f> dst = {
        {0, 0}, {OUT_W - 1.0f, 0}, {OUT_W - 1.0f, OUT_H - 1.0f}, {0, OUT_H - 1.0f}
    };
    int frames = 0;
    double tLast = (double)cv::getTickCount(), fps = 0.0;

    while (true) {
        cv::Mat src;
        if (useCamera) cap >> src;
        else           src = cv::imread(argv[1]);
        if (src.empty()) { if (useCamera) continue; std::cerr << "读不到图像\n"; return 1; }

        // ---- ① 缩放到处理分辨率（离线图片 800，摄像头 640）----
        const int procW = useCamera ? PROC_W : std::min(PROC_W_IMG, src.cols);
        const double scale = (double)procW / src.cols;
        cv::Mat proc;
        cv::resize(src, proc, cv::Size(), scale, scale,
                   src.cols > procW ? cv::INTER_AREA : cv::INTER_CUBIC);
        const double frameArea = (double)proc.rows * proc.cols;

        // ---- ② 中值滤波（压制桌面/布料编织纹理产生的高频假边缘）----
        cv::Mat blur;
        cv::medianBlur(proc, blur, MEDIAN_K);

        // ---- ③ 颜色线索：HSV 阈值切分 + 形态学 ----
        cv::Mat hsv, maskColor;
        cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, LO, HI, maskColor);
        cv::morphologyEx(maskColor, maskColor, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
        cv::morphologyEx(maskColor, maskColor, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, {COLOR_CLOSE, COLOR_CLOSE}));

        // ---- ④ 边缘线索：灰度 Canny + Lab 色度梯度 ----
        // 保留原始灰度算梯度：CLAHE 会放大卡面纹理，导致"边/内比"被拉低而误否真卡片
        cv::Mat grayRaw;
        cv::cvtColor(blur, grayRaw, cv::COLOR_BGR2GRAY);
        cv::Mat gray = grayRaw.clone();
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray, gray);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
        cv::Mat edgesGray;
        cv::Canny(gray, edgesGray, 50, 150);

        cv::Mat lab, chans[3];
        cv::cvtColor(blur, lab, cv::COLOR_BGR2Lab);
        cv::split(lab, chans);
        cv::Mat mag = cv::Mat::zeros(lab.size(), CV_32F);
        for (int k = 1; k <= 2; ++k) {
            cv::Mat ch, sx, sy, m;
            cv::GaussianBlur(chans[k], ch, cv::Size(5, 5), 0);
            ch.convertTo(ch, CV_32F);
            cv::Sobel(ch, sx, CV_32F, 1, 0, 3);
            cv::Sobel(ch, sy, CV_32F, 0, 1, 3);
            cv::magnitude(sx, sy, m);
            mag = cv::max(mag, m);
        }
        cv::Mat mag8, edgesChroma;
        mag.convertTo(mag8, CV_8U, 0.25);
        cv::threshold(mag8, edgesChroma, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        cv::Mat edges;
        cv::bitwise_or(edgesGray, edgesChroma, edges);
        cv::morphologyEx(edges, edges, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));

        // ---- ⑤ 梯度幅值图（给边缘支撑度校验用，用原始灰度）----
        cv::Mat gx, gy, grad;
        cv::Sobel(grayRaw, gx, CV_32F, 1, 0, 3);
        cv::Sobel(grayRaw, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, grad);

        // ---- ⑥ 两条线索各自收集候选 → 打分取最优 ----
        std::vector<Candidate> cands;
        collect(maskColor, grad, frameArea, true, cands);
        collect(edges, grad, frameArea, false, cands);

        cv::Mat warped;
        bool found = !cands.empty();
        std::string info;

        if (found) {
            auto best = *std::max_element(cands.begin(), cands.end(),
                                          [](const Candidate& a, const Candidate& b) {
                                              return a.score < b.score;
                                          });
            // 自动转正：卡片在图像里"竖着"时，四角顺序滚一位，让长边对上输出的长边
            double topLen = cv::norm(best.quad[1] - best.quad[0]);
            double leftLen = cv::norm(best.quad[3] - best.quad[0]);
            if (leftLen > topLen)
                best.quad = {best.quad[1], best.quad[2], best.quad[3], best.quad[0]};

            // ---- ⑦ 透视校正：四点 → 四点，解单应矩阵后重采样 ----
            cv::Mat M = cv::getPerspectiveTransform(best.quad, dst);
            cv::warpPerspective(proc, warped, M, cv::Size(OUT_W, OUT_H), cv::INTER_LINEAR);

            // ---- ⑧ 画绿框 + "ID Card" 标签（对齐参考效果图）----
            std::vector<cv::Point> qi;
            for (const auto& p : best.quad) qi.emplace_back(cvRound(p.x), cvRound(p.y));
            cv::polylines(proc, qi, true, cv::Scalar(0, 255, 0), 2);
            for (int k = 0; k < 4; ++k) cv::circle(proc, qi[k], 5, cv::Scalar(0, 0, 255), -1);
            int lx = std::max(2, qi[0].x), ly = std::max(24, qi[0].y - 8);
            cv::putText(proc, "ID Card", cv::Point(lx, ly),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

            char buf[160];
            snprintf(buf, sizeof(buf), "Detected | AR=%.2f | Area=%.1f%%",
                     best.ar, best.areaPct);
            info = buf;
            if (!useCamera) {
                std::cout << "✅ 检测到卡片：来源=" << best.source
                          << "，面积=" << best.areaPct << "%，长宽比=" << best.ar
                          << "，矩形度=" << best.rect << std::endl;
            }
        } else {
            info = "Not detected";
        }

        // ---- ⑨ 显示 ----
        if (useCamera) {
            int W = proc.cols + 20 + OUT_W;
            int H = std::max(proc.rows + 26, OUT_H + 40);
            cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(28, 28, 28));
            proc.copyTo(canvas(cv::Rect(0, 0, proc.cols, proc.rows)));

            cv::putText(canvas, "Perspective Result", cv::Point(proc.cols + 22, 24),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            if (found) {
                warped.copyTo(canvas(cv::Rect(proc.cols + 20, 34, OUT_W, OUT_H)));
                cv::rectangle(canvas, cv::Rect(proc.cols + 19, 33, OUT_W + 1, OUT_H + 1),
                              cv::Scalar(0, 255, 0), 1);
            } else {
                cv::rectangle(canvas, cv::Rect(proc.cols + 20, 34, OUT_W, OUT_H),
                              cv::Scalar(90, 90, 90), 1);
                cv::putText(canvas, "waiting for card ...",
                            cv::Point(proc.cols + 110, 34 + OUT_H / 2),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(120, 120, 120), 1);
            }

            ++frames;
            double now = (double)cv::getTickCount();
            double dt = (now - tLast) / cv::getTickFrequency();
            if (dt > 0) fps = 0.8 * fps + 0.2 / dt;
            tLast = now;

            char sbuf[200];
            snprintf(sbuf, sizeof(sbuf), "FPS %.1f | %s", fps, info.c_str());
            cv::putText(canvas, sbuf, cv::Point(6, proc.rows + 18),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 180, 255), 1);

            cv::imshow("Student Card Detect", canvas);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') break;                       // ESC / q 退出
            if (key == 's' && found) {                                 // s 保存截图
                cv::imwrite("shot_detect.png", proc);
                cv::imwrite("shot_card.png", warped);
                std::cout << "已保存 shot_detect.png / shot_card.png" << std::endl;
            }
        } else {
            std::string prefix = (argc > 2) ? argv[2] : std::string("result");
            cv::imwrite(prefix + "_detect.png", proc);
            if (found) cv::imwrite(prefix + "_card.png", warped);
            std::cout << "已保存 " << prefix << "_detect.png" << std::endl;
            if (!std::getenv("NO_WINDOW")) {
                cv::imshow("detect", proc);
                if (found) cv::imshow("card", warped);
                cv::waitKey(0);
            }
            break;
        }
    }
    if (cap.isOpened()) cap.release();
    cv::destroyAllWindows();
    return 0;
}
