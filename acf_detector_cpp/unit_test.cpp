#ifdef _DEBUG
#include "acf.h"

using namespace std;
using namespace cv;

static const string train_pos_dir = "E:\\data\\INRIAPerson\\train_64x128_H96\\pos";
static const string test_pos_dir = "E:\\data\\INRIAPerson\\Test\\pos";

void test_bgr2acf()
{
	Mat img = imread(train_pos_dir + "\\crop_000010a.png");
	if (img.empty())
	{
		cerr << "Can not load image " << endl;
		system("pause");
		return;
	}
	imshow("ImageShow", img);
	waitKey(0);

	clock_t start = clock();
	Mat img_crop = img(Rect(16, 16, 64, 128));
	Mat acf = bgr2acf(img_crop);
	cout << acf.size << endl;
	clock_t end = clock();
	cout << "spent time: " << (end - start) << "ms " << endl;
	vector<Mat> channels;
	split(acf, channels);
	for (int i = 0; i < channels.size(); i++)
	{
		double max_v, min_v;
		minMaxLoc(channels[i], &min_v, &max_v);
		cout << "min: " << min_v << " max: " << max_v << endl;
		multiply(channels[i], 16, channels[i]);
		channels[i].convertTo(channels[i], CV_8U);
		imshow("ImageShow" + to_string(i + 2), channels[i]);
		waitKey(0);
	}

	destroyAllWindows();
}

void test_bgr2luv()
{
	Mat img = imread(test_pos_dir + "\\crop001639.png");
	if (img.empty())
	{
		cerr << "Can not load image " << endl;
		system("pause");
		return;
	}
	imshow("ImageShow", img);
	waitKey(0);
	int rows = img.rows;
	int cols = img.cols;
	img.convertTo(img, CV_32F);
	Mat img_luv(img.size(), img.type());
	bgr2luv(img.ptr<float>(0), img_luv.ptr<float>(0), rows*cols, 1.f / 255.f);

	//print
	img_luv *= 255;
	img_luv.convertTo(img_luv, CV_8U);
	vector<Mat> channels(3);
	split(img_luv, channels);
	for (int i = 0; i < channels.size(); ++i)
	{
		imshow("ImageShow", channels[i]);
		waitKey(0);
	}
}

void test_list_dir()
{
	vector<string> file_list;
	list_dir(test_pos_dir, file_list);
	for (vector<string>::const_iterator iter = file_list.begin(); iter != file_list.end(); iter++)
	{
		cout << *iter << endl;
	}
}

void test_mat_write_read()
{
	Mat img = imread(test_pos_dir + "\\crop001639.png");
	if (img.empty())
	{
		cerr << "Can not load image " << endl;
		system("pause");
		return;
	}
	imshow("ImageShow", img);
	waitKey(0);
	matwrite("test.dat", img);
	Mat img_read = matread("test.dat");
	imshow("ImageShow", img_read);
	waitKey(0);
}

void test_get_random()
{
	for (int i = 0; i < 100; ++i)
	{
		cout << get_random() << endl;
	}
}

void test_random_crop()
{
	Mat img = imread(test_pos_dir + "\\crop001639.png");
	if (img.empty())
	{
		cerr << "Can not load image " << endl;
		system("pause");
		return;
	}
	imshow("ImageShow", img);
	waitKey(0);
	for (int i = 0; i < 10; ++i)
	{
		Mat roi = random_crop(img);
		imshow("ImageShow", roi);
		waitKey(0);
	}
}

void test_detect()
{
	vector<string> file_list;
	list_dir(test_pos_dir, file_list);

	//random choose img
	int index = std::abs(get_random()) % file_list.size();
	string file_name = test_pos_dir + "\\" + file_list[index];
	Mat img = imread(file_name);
	imshow("img", img);
	waitKey();

	//detect
	Ptr<ml::Boost> clf = ml::Boost::load("clf.02.yml.bak");
	clock_t start = clock();
	Mat bbox_list = detect(img, clf);
	clock_t end = clock();
	cout << "detection time: " << (end - start) << "ms" << endl;

	//draw boxes
	Mat img_copy = img.clone();
	draw_bbox(img_copy, bbox_list);

	imshow("detect", img_copy);
	waitKey();

	start = clock();
	bbox_list = non_max_suppression(bbox_list, 0.5f);
	end = clock();
	cout << "nms time: " << (end - start) << "ms" << endl;

	//draw boxes
	img_copy = img.clone();
	draw_bbox(img_copy, bbox_list);

	imshow("nms", img_copy);
	waitKey();

	destroyAllWindows();
}

#endif