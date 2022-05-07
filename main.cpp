#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include "aipcpp/face.h"
using namespace cv;
using namespace std;
using namespace aip;

int main(int, char**)
{
    
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Camera open failed" <<endl;
        return -1;
    }
    cerr << "Camera open success" <<endl;

    Mat ColorImage;  //彩色图
    Mat GrayImage;   //灰度图
    //级联分类器
    CascadeClassifier Classifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml");  //加载人脸检测模型
    vector<Rect>  Allface;
    vector<uchar> JPGFace; //JPG人脸容器
    Mat MatFace; //人脸容器
    // 设置APPID/AK/SK
    std::string app_id  =  "xxx";
    std::string api_key =  "xxx";
    std::string secret_key = "xxx";

    aip::Face client(app_id, api_key, secret_key);
    Json::Value result;
    string Base64face;
    time_t new_time;

    while(1)
    {
        cap >> ColorImage;
        //1.彩色图转换成灰度图
        cvtColor(ColorImage,GrayImage,CV_BGR2GRAY);  //彩色图转换成灰度图
        //2.直方图均衡化
        equalizeHist(GrayImage,GrayImage);
        //3.从图像中检测人脸并获得人脸坐标信息
        Classifier.detectMultiScale(GrayImage,Allface);
        //4.画出检测的人脸
        if(Allface.size()){
             rectangle(ColorImage,Allface[0],Scalar(255,255,255)); //方框颜色 BGR
             MatFace = GrayImage(Allface[0]); //截取人脸
             imencode(".jpg",MatFace,JPGFace);
           
             Base64face = base64_encode((char *)JPGFace.data(),JPGFace.size());

             result = client.search(Base64face, "BASE64","demo", aip::null);

             if(!result["result"].isNull()){  //判断数据是否有人脸
                   if(result["result"]["user_list"][0]["score"].asInt() > 70){  //判断对比数据
                            String str= result["result"]["user_list"][0]["user_id"].asString();
                            if(str.compare("xiaozhi")==0){
                                cout<<"识别成功: xiaozhi"<<endl;
                                //在图片中输出文字
                                putText(ColorImage,str,Point(0,50),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
                            }
                            if(str.compare("xiaoming")==0){
                                cout<<"识别成功: xiaoming"<<endl;
                                //在图片中输出文字
                                putText(ColorImage,str,Point(0,100),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
                            }
                            time(&new_time);
                            cout<<ctime(&new_time)<<endl;
                   }else{
                       //cout<<"识别失败"<<endl;
                   }
             }
        }
        imshow("video", ColorImage);
        waitKey(40);  //40dps
    }

    return 0;
}