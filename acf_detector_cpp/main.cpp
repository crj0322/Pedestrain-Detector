#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include "acf.h"

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
	//generate_pos_data("E:\\data\\INRIAPerson\\train_64x128_H96\\pos", "X_pos.dat");
	//generate_neg_data("E:\\data\\INRIAPerson\\train_64x128_H96\\neg", "X_neg.dat");
	//generate_pos_data("E:\\data\\INRIAPerson\\test_64x128_H96\\pos", "X_pos_test.dat");
	//generate_neg_data("E:\\data\\INRIAPerson\\test_64x128_H96\\neg", "X_neg_test.dat");
	//train("clf.01.yml");
	//evaluate("clf.01.yml");
	//generate_hard_neg("E:\\data\\INRIAPerson\\train_64x128_H96\\neg", "X_hard_neg.dat", "clf.01.yml");
	train("clf.02.yml");
	//evaluate("clf.02.yml");
	test_detect();

	system("pause");

	return 0;
}