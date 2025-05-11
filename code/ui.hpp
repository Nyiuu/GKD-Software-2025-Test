#include <thread>
#include <mutex>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "modelbase.hpp"

using namespace cv;
using namespace std;

template<typename T>
class UI{
private:
    Mat g_canvas;
    Mat g_display;
    vector<float> g_probs;
    mutex g_mutex;
    bool g_drawing;
    Point g_last_point;
    unique_ptr<ModelBase> model;
    string folder_path;

    
public:
    UI(const string& path, unique_ptr<ModelBase> model_ptr):g_canvas(400, 400, CV_8UC1, Scalar(255)), g_display(450, 650, CV_8UC3, Scalar(255, 255, 255)), 
    g_probs(10, 0.0f), g_drawing(false), folder_path(path), model(move(model_ptr)){
        model->load_model(folder_path);
    }

    ~UI(){

    }

    static void* recognition_thread_static(void* arg) {
        UI* self = static_cast<UI*>(arg);
        self->recognition_thread();
        return nullptr;
    }

    void recognition_thread(){
        while (true){
            
            Mat canvas_copy;
            {
                lock_guard<mutex> lock(g_mutex);
                canvas_copy = g_canvas.clone();
            }//准确获取画布内容
            
            //开始画
            if(countNonZero(canvas_copy < 255) > 0){
                    string temp_path = "../temp/temp_digit.png";
                    imwrite(temp_path, canvas_copy);

                    auto result = model->forward(temp_path);
                    
                    //test
                    // auto start = std::chrono::high_resolution_clock::now();
                    // auto result = model.forward(temp_path);
                    // auto end = std::chrono::high_resolution_clock::now();
                    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    // std::cout << "耗时: " << duration.count() / 1000000.0 << " 秒" << std::endl;
                    //test

                    if(!result.empty()){
                        lock_guard<mutex> lock(g_mutex);
                        for(int i = 0; i < 10; ++i){
                            g_probs[i] = result[i];
                        }
                    }
            }else{
                // 清空概率
                lock_guard<mutex> lock(g_mutex);
                fill(g_probs.begin(), g_probs.end(), 0.0f);
            }
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }

    static void onMouseStatic(int event, int x, int y, int flags, void* userdata) {
        UI* self = static_cast<UI*>(userdata);
        self->on_mouse(event, x, y, flags);
    }

    void on_mouse(int event, int x, int y, int flags){
        if (x < 0 || x >= g_canvas.cols || y < 0 || y >= g_canvas.rows){
            return;
        }
        
        if (event == EVENT_LBUTTONDOWN) {
            g_drawing = true;
            g_last_point = Point(x, y);
        } else if (event == EVENT_LBUTTONUP) {
            g_drawing = false;
        } else if (event == EVENT_MOUSEMOVE && g_drawing) {
            lock_guard<mutex> lock(g_mutex);
            line(g_canvas, g_last_point, Point(x, y), Scalar(0), 10, LINE_AA);
            g_last_point = Point(x, y);
        }
    }

    void draw_probabilities(Mat& display, const vector<float>& probs) {
        int graph_width = 150;
        int graph_height = 300;
        int graph_x = 420;
        int graph_y = 50;
        int bar_width = 10;
        int spacing = 5;
        
        // 绘制背景
        rectangle(display, Rect(graph_x, graph_y, graph_width, graph_height), 
                    Scalar(200, 200, 200), FILLED);
        
        // 绘制每个数字的概率条
        for (int i = 0; i < 10; ++i) {
            int bar_x = graph_x + spacing + i * (bar_width + spacing);
            int bar_height = static_cast<int>(probs[i] * graph_height);
            
            // 绘制柱状图
            rectangle(display, 
                        Rect(bar_x, graph_y + graph_height - bar_height, 
                                bar_width, bar_height), 
                        Scalar(100, 100, 255), FILLED);
            
            // 绘制数字标签
            putText(display, to_string(i), 
                        Point(bar_x, graph_y + graph_height + 20), 
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
            
            // 显示概率值
            string prob_text = format("%.2f", probs[i]);
            putText(display, prob_text, 
                        Point(bar_x - 5, graph_y + graph_height - bar_height - 5), 
                        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
        }
    }

    void ui_task(){
        
        // 创建窗口
        namedWindow("Digit Recognizer", WINDOW_AUTOSIZE);
        setMouseCallback("Digit Recognizer", &UI::onMouseStatic, this);
        
        // 启动识别线程
        thread rec_thread(&recognition_thread_static, this);
        rec_thread.detach();

        test();

        // 清空按钮
        Mat button(50, 100, CV_8UC3, Scalar(200, 200, 200));
        putText(button, "Clear", Point(10, 30), 
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);
        
         // 主循环
        while (true) {
            // 创建显示图像
            {
                lock_guard<mutex> lock(g_mutex);
                
                // 清空显示
                g_display.setTo(Scalar(255, 255, 255));
                
                // 绘制画布内容
                Mat roi = g_display(Rect(10, 10, g_canvas.cols, g_canvas.rows));
                cvtColor(g_canvas, roi, COLOR_GRAY2BGR);
                
                // 绘制概率柱状图
                draw_probabilities(g_display, g_probs);
                
                // 绘制识别结果
                if (!all_of(g_probs.begin(), g_probs.end(), [](float p) { return p == 0.0f; })) {
                    auto max_it = max_element(g_probs.begin(), g_probs.end());
                    int predicted = static_cast<int>(max_it - g_probs.begin());
                    putText(g_display, "Predicted: " + to_string(predicted), 
                                Point(420, 30), FONT_HERSHEY_SIMPLEX, 0.8, 
                                Scalar(0, 0, 255), 2);
                }
                
                // 添加清空按钮
                button.copyTo(g_display(Rect(250, 350, button.cols, button.rows)));
            }
            
            // 显示图像
            imshow("Digit Recognizer", g_display);
            
            // 检查按钮点击
            int key = waitKey(30);
            if (key == 27) { // ESC退出
                break;
            } else if (key == 'c' || key == 'C') {
                // 清空画布
                lock_guard<mutex> lock(g_mutex);
                g_canvas.setTo(Scalar(255));
            }
        }
    }

    void test(){
        model->process_all();
    }

};


