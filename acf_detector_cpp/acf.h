#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

#define PI 3.14159265358979323846f

//acf.cpp

cv::Mat bgr2acf(cv::Mat& img, int pool_size=4);

template<class oT> oT* rgb2luv_setup(oT z, oT *mr, oT *mg, oT *mb,
	oT &minu, oT &minv, oT &un, oT &vn);

template<class iT, class oT> void bgr2luv(iT *I, oT *J, int n, oT nrm);

//end acf.cpp

//utils.cpp

void list_dir(const std::string& name, std::vector<std::string>& v);

void matwrite(const std::string& filename, const cv::Mat& mat);

cv::Mat matread(const std::string& filename);

int get_random();

//end utils.cpp

//train.cpp

void generate_pos_data(std::string data_dir, std::string file_name, cv::Size data_shape=cv::Size(64, 128));

cv::Mat random_crop(cv::Mat img, cv::Size crop_shape=cv::Size(64, 128));

std::vector<cv::Mat> get_img_list(std::string data_dir);

void generate_neg_data(std::string data_dir, std::string file_name, cv::Size data_shape=cv::Size(64, 128), int sample_size=5000);

void generate_hard_neg(std::string data_dir, std::string file_name, std::string clf_path,
	cv::Size data_shape=cv::Size(64, 128), int sample_size=5000, int batch_size=500);

void train(std::string file_name);

void evaluate(std::string clf_path);

//end train.cpp

//detect.cpp

cv::Mat detect(cv::Mat img, cv::Ptr<cv::ml::Boost> clf,
	cv::Size window_wh= cv::Size(64, 128), int pool_size=4, int stride=1);

cv::Mat slice_row(cv::Mat input, std::vector<int> idx);

cv::Mat non_max_suppression(cv::Mat bbox_list, float iou_threshold);

void draw_bbox(cv::Mat& img, cv::Mat bbox_list);

//end detect.cpp

#ifdef _DEBUG
//unit_test.cpp

void test_bgr2acf();

void test_bgr2luv();

void test_list_dir();

void test_mat_write_read();

void test_get_random();

void test_random_crop();

void test_detect();

//end unit_test.cpp
#endif