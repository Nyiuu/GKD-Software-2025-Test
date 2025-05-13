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
using namespace std;

mutex coutMutex;

struct MatrixData{
    string type;
    string matrix_name;
    int row;
    int col;
};

void socket_task(){
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
    s_stream << "lalala";
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
    char buffer[1024] = {0};
    int bytesReceived = recv(sock, buffer, sizeof(buffer), 0);
    {
        lock_guard<mutex> lock(coutMutex);
        if (bytesReceived <= 0) {
            cerr << "未收到服务器响应" << endl;
        } else {
           cout << "OK" << endl; 
        }
    }

    shutdown(sock, SHUT_RDWR);
    close(sock);
}

int main() {
    vector<thread> threads;

    threads.emplace_back(socket_task);

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}