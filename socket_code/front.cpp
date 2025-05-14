#include <string>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <mutex>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <thread>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
mutex coutMutex;

template<typename T>
void load_data(stringstream& buffer_stream, const vector<vector<T>>& matrix){
    int row = matrix.size();
    int col = matrix[0].size();
    for(int i = 0; i < row; ++i){
        for(int j = 0; j < col; ++j){
            buffer_stream << matrix[i][j];
            if (j != col - 1) {
                buffer_stream << " "; 
            }
        }
        buffer_stream << "\n";
    }
}

template<typename T>
vector<vector<T>> processImage(const string& imagePath) {
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    
    if(image.empty()){
        cerr << "无法加载图像: " << imagePath << endl;
        return {};
    }

    Mat resizedImage;
    resize(image, resizedImage, Size(28, 28), 0, 0, INTER_LINEAR);

    Mat flattened = resizedImage.reshape(1, 1);
    if constexpr(is_same_v<T, float>){
        flattened.convertTo(flattened, CV_32F); 
    }else if constexpr(is_same_v<T, double>){
        flattened.convertTo(flattened, CV_64F);  
    }

    vector<T> normalizedVec;
    flattened /= 255.0;
    normalizedVec.assign(flattened.begin<T>(), flattened.end<T>());
    vector<vector<T>> result;
    result.push_back(normalizedVec);
    return result;
}

template<typename T>
void load_all(stringstream& buffer_stream){
    const string folderPath = "../nums/";
    const int numImages = 10;
    for (int i = 0; i < numImages; ++i) {
        string imagePath = folderPath + to_string(i) + ".png"; 
        auto matrix = processImage<T>(imagePath);
        load_data(buffer_stream, matrix);
        buffer_stream << "E\n";
    }
}



template<typename T>
void socket_task(const string&choose){
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    
    // 创建socket
    if (sock < 0) {
        lock_guard<mutex> lock(coutMutex);
        cerr << "创建套接字失败" << endl;
        return;
    }
     //关闭端口  
     int reuse = 1;
     if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
         shutdown(sock, SHUT_RDWR);
         close(sock);       
         return;
     }
     // 设置连接超时（5秒）
     timeval timeout{5, 0};
     setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
     setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    // 连接服务器
    sockaddr_in servAddr{};
    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(8080);
    inet_pton(AF_INET, "127.0.0.1", &servAddr.sin_addr);
   
    if (connect(sock, (struct sockaddr*)&servAddr, sizeof(servAddr)) < 0) {
        lock_guard<mutex> lock(coutMutex);
        cerr << "连接服务器失败" << endl;
        shutdown(sock, SHUT_RDWR);
        close(sock);
        return;
    }

    stringstream s_stream;

    s_stream << choose + "\n";
    load_all<T>(s_stream);
    string str = s_stream.str();

    // 发送
    if (send(sock, str.c_str(), str.size(), 0) < 0) {
        lock_guard<mutex> lock(coutMutex);
        cerr << "发送失败" << endl;
        shutdown(sock, SHUT_RDWR);
        close(sock);
        return;
    }

    // 接收响应
    char buffer[10240] = {0};
    int bytesReceived = recv(sock, buffer, sizeof(buffer), 0);
    {
        lock_guard<mutex> lock(coutMutex);
        if (bytesReceived <= 0) {
            cerr << "未收到服务器响应" << endl;
        } else {
           cout << buffer << endl;
        }
    }

    shutdown(sock, SHUT_RDWR);
    close(sock);
}

int main() {
    vector<thread> threads;
    cout << "choose your model type:1 for mnist-fc, 2 for mnist-fc-plus" << endl;
    string choose;
    cin >> choose;
    bool success = true;
    if(choose == "1"){
        threads.emplace_back(socket_task<float>, choose);
    }else if(choose == "2"){
        threads.emplace_back(socket_task<double>, choose);
    }else{
        success = false;
    }
    
    if(success){
        for (auto& t : threads) {
            t.join();
        }
    }
    return 0;
}