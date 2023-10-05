#pragma once
#ifndef ARMOR_H
#define ARMOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "LightBar.hpp"


using namespace std;
using namespace cv;
using namespace dnn;

class Armor {
private:
	Mat grayimg;
	Net net = readNetFromONNX("../config/model.onnx");
	LightBar lightbar1,lightbar2;
	Mat imgwarp;


	Mat normalize(const Mat& inImage) {
    	double minVal, maxVal;
		float mean_value = 0.458;
		float std_value = 0.226;
    	Mat Image;
    	inImage.convertTo(Image, CV_32FC1, 1.0   / (std_value * 255), (0.0 - mean_value) / std_value);
    	return Image;
	}

	int predict(){
		cvtColor(imgwarp, grayimg, COLOR_BGR2GRAY);
		grayimg = normalize(grayimg);
		Mat resizedimg;
		//cv::threshold(grayimg, grayimg, 127, 255, THRESH_BINARY);
		// imshow("put_in", grayimg);
		resize(grayimg,resizedimg,Size(48,36));
		Mat blob = blobFromImage(resizedimg);
		net.setInput(blob);
		Mat probs = net.forward();
		int bestClass = 0;
		double bestClassProb = 0.0;
		for (int i = 0; i < probs.cols; ++i) {
    		double prob = probs.at<float>(0, i); 
    		if (prob > bestClassProb) {
        		bestClass = i;
        		bestClassProb = prob;
    		}
		}
		cout << bestClass <<endl;
		return bestClass;
	}

	double Area(const Point2f& A, const Point2f& B, const Point2f& C) {
    double side1 = norm(A - B);
    double side2 = norm(B - C);
    double side3 = norm(C - A);
    double s = (side1 + side2 + side3) / 2.0;
    double area = sqrt(s * (s - side1) * (s - side2) * (s - side3));   
    return area*2;
}
	
public:
	bool isEmpty = true;

	int result = 0;
	int targetnumber = 0;

	bool tracktarget = false;

	LightBar get_lightbar1(){
		return lightbar1;
	}

	LightBar get_lightbar2(){
		return lightbar2;
	}

	Armor(LightBar in1,LightBar in2,Point2f centre = Point2f(0,0)){
		lightbar1 = in1;
		lightbar2 = in2;

	}

	vector<Point2f> getPointsAlter(double k = 2){  
		Point2f E,F,G,H;  
		Point2f mid1 = (lightbar1.get_middle_point(lightbar1.get_lightBar())[0]+lightbar1.get_middle_point(lightbar1.get_lightBar())[1])/2;
		Point2f mid2 = (lightbar2.get_middle_point(lightbar2.get_lightBar())[0]+lightbar2.get_middle_point(lightbar2.get_lightBar())[1])/2;
		if(mid1.x<=mid2.x){
			E = lightbar1.get_middle_point(lightbar1.get_lightBar())[0];
			H = lightbar1.get_middle_point(lightbar1.get_lightBar())[1];
			F = lightbar2.get_middle_point(lightbar2.get_lightBar())[0];
			G = lightbar2.get_middle_point(lightbar2.get_lightBar())[1];
		}
		else{
			E = lightbar2.get_middle_point(lightbar2.get_lightBar())[0];
			H = lightbar2.get_middle_point(lightbar2.get_lightBar())[1];
			F = lightbar1.get_middle_point(lightbar1.get_lightBar())[0];
			G = lightbar1.get_middle_point(lightbar1.get_lightBar())[1];
		}
		vector<Point2f> return_Points = {E,F,G,H};
		return return_Points;
	}

	vector<Point2f> getPoints(double k = 2.2){  
		Point2f A,B,C,D;
		Point2f E,F,G,H;  
		Point2f mid1 = (lightbar1.get_middle_point(lightbar1.get_lightBar())[0]+lightbar1.get_middle_point(lightbar1.get_lightBar())[1])/2;
		Point2f mid2 = (lightbar2.get_middle_point(lightbar2.get_lightBar())[0]+lightbar2.get_middle_point(lightbar2.get_lightBar())[1])/2;
		if(mid1.x<=mid2.x){
			E = lightbar1.get_middle_point(lightbar1.get_lightBar())[0];
			H = lightbar1.get_middle_point(lightbar1.get_lightBar())[1];
			F = lightbar2.get_middle_point(lightbar2.get_lightBar())[0];
			G = lightbar2.get_middle_point(lightbar2.get_lightBar())[1];
		}
		else{
			E = lightbar2.get_middle_point(lightbar2.get_lightBar())[0];
			H = lightbar2.get_middle_point(lightbar2.get_lightBar())[1];
			F = lightbar1.get_middle_point(lightbar1.get_lightBar())[0];
			G = lightbar1.get_middle_point(lightbar1.get_lightBar())[1];
		}
		A = E+(E-H)/k;
		B = F+(F-G)/k;
		C = G+(G-F)/k;
		D = H+(H-E)/k;
		vector<Point2f> return_Points = {A,B,C,D};
		return return_Points;
	}
	
	Point2f center(){
		vector<Point2f> points = getPoints();
		return (points[0]+points[1]+points[2]+points[3])/4;
		
	}

	int get_number(){
		result = predict();
		return result;
	}

	void get_number_Image(Mat input,int k = -10){
		vector<Point2f> Points = getPoints();
		Point2f src[4] = {Points[0]+(Points[1]-Points[0])/k,Points[1]-(Points[1]-Points[0])/k,Points[2]+(Points[3]-Points[2])/k,Points[3]-(Points[3]-Points[2])/k};
		//Point2f src[4] = {Points[0],Points[1],Points[2],Points[3]};
		// cout << "test" << endl;
		// cout << "points1:"<< Points[0].x << "," << Points[0].y <<
		// "points2:"<< Points[1].x << "," << Points[1].y <<
		// "points3:"<< Points[2].x << "," << Points[2].y <<
		// "points4:"<< Points[3].x << "," << Points[3].y <<endl;
		// cout << "test2" << endl;
		// cout << "points1:"<< (Points[0]+(Points[1]-Points[0])/k).x << "," << (Points[0]+(Points[1]-Points[0])/k).y <<
		// "points2:"<< (Points[1]-(Points[1]-Points[0])/k).x << "," << (Points[1]-(Points[1]-Points[0])/k).y <<
		// "points3:"<< (Points[2]+(Points[3]-Points[2])/k).x << "," << (Points[2]+(Points[3]-Points[2])/k).y <<
		// "points4:"<<(Points[3]-(Points[3]-Points[2])/k).x << "," << (Points[3]-(Points[3]-Points[2])/k).y <<endl;
		
		Point2f dst[4] = {{0.0,0.0},{640.0,0.0},{640.0,480.0},{0.0,480.0}};
		Mat matrix = getPerspectiveTransform(src,dst);
		warpPerspective(input,imgwarp,matrix,Point(640,480));
		// imshow("warp",imgwarp);
	}

	double Score(){
		double area = Area(lightbar1.get_middle_point(lightbar1.get_lightBar())[0],lightbar1.get_middle_point(lightbar1.get_lightBar())[1],lightbar2.get_middle_point(lightbar2.get_lightBar())[0]);
		double delta_slope = lightbar1.get_slope(lightbar1.get_lightBar()) - lightbar2.get_slope(lightbar2.get_lightBar());
		double delta_area = lightbar1.get_area(lightbar1.get_lightBar()) / lightbar2.get_area(lightbar2.get_lightBar());
		double rectness = (area/(norm(lightbar1.get_middle_point(lightbar1.get_lightBar())[0]-lightbar1.get_middle_point(lightbar1.get_lightBar())[1])*norm(lightbar1.get_middle_point(lightbar1.get_lightBar())[0]-lightbar2.get_middle_point(lightbar2.get_lightBar())[0])));
		double aspect_ratio = 1/(norm(lightbar1.get_middle_point(lightbar1.get_lightBar())[0]-lightbar1.get_middle_point(lightbar1.get_lightBar())[1])/norm(lightbar1.get_middle_point(lightbar1.get_lightBar())[0]-lightbar2.get_middle_point(lightbar2.get_lightBar())[0]));
		double dialength_ratio =(norm(lightbar1.get_middle_point(lightbar1.get_lightBar())[0]-lightbar2.get_middle_point(lightbar2.get_lightBar())[1])/norm(lightbar1.get_middle_point(lightbar1.get_lightBar())[1]-lightbar2.get_middle_point(lightbar2.get_lightBar())[0]))<1?
		(norm(lightbar1.get_middle_point(lightbar1.get_lightBar())[0]-lightbar2.get_middle_point(lightbar2.get_lightBar())[1])/norm(lightbar1.get_middle_point(lightbar1.get_lightBar())[1]-lightbar2.get_middle_point(lightbar2.get_lightBar())[0])):
		1/(norm(lightbar1.get_middle_point(lightbar1.get_lightBar())[0]-lightbar2.get_middle_point(lightbar2.get_lightBar())[1])/norm(lightbar1.get_middle_point(lightbar1.get_lightBar())[1]-lightbar2.get_middle_point(lightbar2.get_lightBar())[0]));
		double score = (delta_area*10 + (1-delta_slope)*90)*rectness*(aspect_ratio<2?1:2)*dialength_ratio*dialength_ratio;
		return score;
	}

};

#endif
