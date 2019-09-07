// face_detection_test1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#if 0


#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

CascadeClassifier face_cascader;
CascadeClassifier eye_cascader;
String facefile = "C:/Program Files/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";//
String eyefile = "C:/Program Files/opencv/build/etc/haarcascades/haarcascade_eye.xml";//

int main(int argc, char** argv) {

	if (!face_cascader.load(facefile)) {	//检测是否读入人脸特征数据
		printf("could not load face feature data...\n");
		return -1;
	}
	if (!eye_cascader.load(eyefile)) {		//检测是否读入人眼特征数据
		printf("could not load eye feature data...\n");
		return -1;
	}

	namedWindow("camera-demo", CV_WINDOW_AUTOSIZE);  //打开一个窗口
	VideoCapture capture(0);                         //调用摄像头(也可以读入视频文件)

	Mat frame;										
	Mat gray;

	vector<Rect> faces;    //定义脸的矩形容器
	vector<Rect> eyes;	   //定义眼的矩形容器

	while (capture.read(frame)) //当读取到帧图像时执行程序
	{
		cvtColor(frame, gray, COLOR_BGR2GRAY);//转换为灰度图像
		equalizeHist(gray, gray);//进行直方图均衡化处理,提升准确度
		face_cascader.detectMultiScale(gray, faces, 1.2, 3, 0, Size(30, 30));
		//printf("detect face number is %d\n", faces.size());
		for (size_t t = 0; t < faces.size(); t++)
		{
			Rect roi;
			roi.x = faces[static_cast<int>(t)].x;//获取脸部的左上角坐标
			roi.y = faces[static_cast<int>(t)].y;
			roi.width = faces[static_cast<int>(t)].width;//获取长度以及宽度信息
			roi.height = faces[static_cast<int>(t)].height /2 ; //除以2为了将眼部的检测范围缩小
			Mat faceROI = frame(roi);
			rectangle(frame, faces[static_cast<int>(t)], Scalar(0, 0, 255), 2, 8, 0);//将脸部用红色矩形标注出来

			eye_cascader.detectMultiScale(faceROI, eyes, 1.2, 3, 0, Size(20, 20));
			//printf("detect eye number is %d\n", eyes.size());
			for (size_t k = 0; k < eyes.size(); k++) 
			{
				Rect rect;
				rect.x = faces[static_cast<int>(t)].x + eyes[k].x;
				rect.y = faces[static_cast<int>(t)].y + eyes[k].y;
				rect.width = eyes[k].width;
				rect.height = eyes[k].height;
				rectangle(frame, rect, Scalar(0, 255, 0), 2, 8, 0);//绿色标注眼睛位置
			}
			
		}
		imshow("camera-demo", frame);

		char c = waitKey(20);//延时20ms
		if (c == 27) {       //按下esc退出
			break;
		}
	}
	//waitKey(0);
	return 0;
}
#endif // 0

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

String facefile = "C:/Program Files/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
String lefteyefile = "C:/Program Files/opencv/build/etc/haarcascades/haarcascade_lefteye_2splits.xml";
String righteyefile = "C:/Program Files/opencv/build/etc/haarcascades/haarcascade_righteye_2splits.xml";
CascadeClassifier face_detector;
CascadeClassifier leftyeye_detector;
CascadeClassifier righteye_detector;

Rect leftEye, rightEye;

void trackEye(Mat& im, Mat& tpl, Rect& rect) //追踪眼部
{
	Mat result;
	int result_cols = im.cols - tpl.cols + 1;//opencv规定的result大小 将模板图像在待检测图像中进行滑动检测相关性
	int result_rows = im.rows - tpl.rows + 1;//

	// 模板匹配
	result.create(result_rows, result_cols, CV_32FC1);//创建新的矩阵 类型为浮点数
	matchTemplate(im, tpl, result, TM_CCORR_NORMED);//模板匹配
	            //输入，模板，result图像,归一化系数(0-1)
	
	// 寻找位置
	double minval, maxval;
	Point minloc, maxloc;
	minMaxLoc(result, &minval, &maxval, &minloc, &maxloc);//找到result数组内的最小最大值以及其所在位置
	if (maxval > 0.75) //若最大值大于0.75则匹配符合
	{
		rect.x = rect.x + maxloc.x;
		rect.y = rect.y + maxloc.y;
	}
	else //未找到则更新为0
	{
		rect.x = rect.y = rect.width = rect.height = 0;
	}
}

int main(int argc, char** argv) 
{
	//特征数据载入以及检错
	if (!face_detector.load(facefile))
	{
		printf("could not load data file...\n");
		return -1;
	}
	if (!leftyeye_detector.load(lefteyefile))
	{
		printf("could not load data file...\n");
		return -1;
	}
	if (!righteye_detector.load(righteyefile)) 
	{
		printf("could not load data file...\n");
		return -1;
	}

	Mat frame;
	//VideoCapture capture(0);
	
	VideoCapture capture;	
	if (!capture.open("C:/Users/25227/Desktop/test2.mp4"))
		return -2;
	
	namedWindow("original-win", CV_WINDOW_AUTOSIZE);
	Mat frame1;
	/*while (capture.read(frame1))
	{
		imshow("original-win", frame1);
	}*/
	namedWindow("demo-win", CV_WINDOW_AUTOSIZE);

	Mat gray;
	vector<Rect> faces;
	vector<Rect> eyes;
	Mat lefttpl, righttpl; // 定义左右眼模板
	
	while (capture.read(frame)) 
	{
		imshow("original-win", frame);
		//flip(frame, frame, 1); //进行图像翻转
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);
		face_detector.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
		for (size_t t = 0; t < faces.size(); t++)
		{
			rectangle(frame, faces[t], Scalar(255, 0, 0), 2, 8, 0);//标记人脸位置

			// 计算 offset ROI
			int offsety = faces[t].height / 4; //设置偏移量
			int offsetx = faces[t].width / 8;
			int eyeheight = faces[t].height / 2 - offsety; //调整眼部的高度与宽度,进一步缩小眼部的检测区域 提高检测速度
			int eyewidth = faces[t].width / 2 - offsetx;

			// 截取左眼区域 左半边部分为左眼所在区域
			Rect leftRect;
			leftRect.x = faces[t].x + offsetx;
			leftRect.y = faces[t].y + offsety;
			leftRect.width = eyewidth;
			leftRect.height = eyeheight;

			Mat leftRoi = gray(leftRect);//类型转换

			// 检测左眼
			leftyeye_detector.detectMultiScale(leftRoi, eyes, 1.1, 1, 0, Size(20, 20));//在leftROI区域检测左眼
			if (lefttpl.empty()) //若模板为空 则生成模板
			{
				if (eyes.size()) //若检测到眼睛 则生成模板
				{
					leftRect = eyes[0] + Point(leftRect.x, leftRect.y);//获取左眼所在位置
					lefttpl = gray(leftRect); //位置信息存入模板
					rectangle(frame, leftRect, Scalar(0, 0, 255), 2, 8, 0);//标记左眼区域
				}
			}
			else
			{
				// 跟踪， 基于模板匹配
				leftEye.x = leftRect.x;
				leftEye.y = leftRect.y;
				trackEye(leftRoi, lefttpl, leftEye);//进行模板跟踪
				if (leftEye.x > 0 && leftEye.y > 0) //若模板追踪成功
				{
					leftEye.width = lefttpl.cols;
					leftEye.height = lefttpl.rows;
					rectangle(frame, leftEye, Scalar(0, 0, 255), 2, 8, 0);
				}
			}

			// 截取右眼区域
			Rect rightRect;
			rightRect.x = faces[t].x + faces[t].width / 2;
			rightRect.y = faces[t].y + offsety;
			rightRect.width = eyewidth;
			rightRect.height = eyeheight;
			Mat rightRoi = gray(rightRect);

			// 检测右眼
			righteye_detector.detectMultiScale(rightRoi, eyes, 1.1, 1, 0, Size(20, 20));
			if (righttpl.empty()) //若无模板 则计算模板
			{
				if (eyes.size())
				{
					rightRect = eyes[0] + Point(rightRect.x, rightRect.y);
					righttpl = gray(rightRect);
					rectangle(frame, rightRect, Scalar(0, 255, 255), 2, 8, 0);
				}
			}
			else //有模板 进行模板匹配
			{
				// 跟踪， 基于模板匹配
				rightEye.x = rightRect.x;
				rightEye.y = rightRect.y;
				trackEye(rightRoi, righttpl, rightEye);
				if (rightEye.x > 0 && rightEye.y > 0)
				{
					rightEye.width = righttpl.cols;
					rightEye.height = righttpl.rows;
					rectangle(frame, rightEye, Scalar(0, 255, 255), 2, 8, 0);
				}
			}
		}
		imshow("demo-win", frame);
		char c = waitKey(30);
		if (c == 27) { // ESC
			break;
		}
	}

	// release resource
	capture.release();
	waitKey(0);
	return 0;
}
/*
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

int main(int argc, const char** argv)
{
	cv::Mat frame;
	// 可从摄像头输入视频流或直接播放视频文件
	cv::VideoCapture capture(0);
	//cv::VideoCapture capture("vedio1.avi");
	double fps;
	char string[10];  // 帧率字符串
	cv::namedWindow("Camera FPS");
	double t = 0;
	while (1)
	{
		t = (double)cv::getTickCount();
		if (cv::waitKey(1) == 1) { break; }
		if (capture.isOpened())
		{
			capture >> frame;
			// getTickcount函数：返回从操作系统启动到当前所经过的毫秒数
			// getTickFrequency函数：返回每秒的计时周期数
			// t为该处代码执行所耗的时间,单位为秒,fps为其倒数
			t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
			fps = 1.0 / t;
			sprintf_s(string, "%.2f", fps);      // 帧率保留两位小数
			std::string fpsString("FPS:");
			fpsString += string;                    // 在"FPS:"后加入帧率数值字符串
			printf("fps: %.2f width:%d height:%d fps:%.2f\n", fps, frame.cols, frame.rows, capture.get(CV_CAP_PROP_FPS));
			// 将帧率信息写在输出帧上
			cv::putText(frame, // 图像矩阵
				fpsString,                  // string型文字内容
				cv::Point(5, 20),           // 文字坐标，以左下角为原点
				cv::FONT_HERSHEY_SIMPLEX,   // 字体类型
				0.5, // 字体大小
				cv::Scalar(0, 0, 0));       // 字体颜色
			cv::imshow("Camera FPS", frame);
			char c = cv::waitKey(30); //延时30毫秒
			// 注：waitKey延时越长 fps越大 出现跳帧 摄像头显示变卡
			if (c == 27) //按ESC键退出
				break;
		}
		else
		{
			std::cout << "No Camera Input!" << std::endl;
			break;
		}
	}
}
*/