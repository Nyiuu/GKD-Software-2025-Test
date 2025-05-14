//服务器端
#include <fstream>
#include <sstream>
#include <map>
#include <mutex>
#include <thread>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "model.hpp"

using namespace std;

mutex cout_mutex;

void handleClient(int clientSocket) {
    char buffer[65536] = {0};
    int bytesRead = recv(clientSocket, buffer, sizeof(buffer), 0);

    if (bytesRead <= 0) {
        close(clientSocket);
        return;
    }

    buffer[bytesRead] = '\0';

    stringstream ss(buffer);

    string choose;
    getline(ss, choose);
    auto model_ptr = create_model(choose);
    model_ptr->load_model(choose);

    auto response = model_ptr->process_all(ss);
    
    send(clientSocket, response.c_str(), response.size(), 0);

    close(clientSocket);
}

void server_init(int serverFd){
    int opt = 1;
    if (setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        cerr << "设置 SO_REUSEADDR 失败" << endl;
        exit(EXIT_FAILURE);
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);

    bind(serverFd, (struct sockaddr*)&address, sizeof(address));
    listen(serverFd, 5);
    cout << "8080端口启动成功" << endl;
}

void server_loop(int serverFd) {
    while (true) {
        sockaddr_in clientAddr{};
        socklen_t addrLen = sizeof(clientAddr);
        int clientSocket = accept(serverFd, (struct sockaddr*)&clientAddr, &addrLen);
        
        if (clientSocket < 0) {
            cerr << "接受失败" << endl;
            continue;
        }
        thread(handleClient, clientSocket).detach(); //分离线程处理客户端请求
    }
}

int main() { 
    // 创建服务器套接字
    int serverFd = socket(AF_INET, SOCK_STREAM, 0);
    server_init(serverFd);
    server_loop(serverFd);
    return 0;
}