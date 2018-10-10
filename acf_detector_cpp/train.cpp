#include "acf.h"
#include <ml.hpp>

using namespace std;
using namespace cv;

void generate_pos_data(string data_dir, string file_name, Size data_shape)
{
	vector<string> file_list;
	list_dir(data_dir, file_list);
	int feature_num = (int)(file_list.size());
	int feature_dim = data_shape.height * data_shape.width * 10 / 16; //5120
	Mat X_pos(feature_num, feature_dim, CV_32F);
	for (int i = 0; i < feature_num; ++i)
	{
		string file_name = data_dir + "\\" + file_list[i];
		Mat img = imread(file_name);
		if (img.empty())
		{
			cerr << "Can not load image " << file_name << endl;
			system("pause");
			return;
		}
		int w_offset = (img.cols - data_shape.width) / 2;
		int h_offset = (img.rows - data_shape.height) / 2;
		Mat roi = img(Rect(w_offset, h_offset, data_shape.width, data_shape.height));
		Mat acf = bgr2acf(roi);
		acf = acf.reshape(1, 1);
		acf.copyTo(X_pos.row(i));

		//print progress
		cout << (100 * i / feature_num) << "%";
		cout << "\b\b\b\b";
	}
	cout << X_pos.size << endl;
	double max_v, min_v;
	minMaxLoc(X_pos, &min_v, &max_v);
	cout << "min: " << min_v << " max: " << max_v << endl;

	matwrite(file_name, X_pos);
}

Mat random_crop(Mat img, Size crop_shape)
{
	Size org_shape = img.size();
	int x = std::abs(get_random()) % (org_shape.width - crop_shape.width);
	int y = std::abs(get_random()) % (org_shape.height - crop_shape.height);
	//cout << x << " " << y << endl;
	Mat roi = img(Rect(x, y, crop_shape.width, crop_shape.height));
	return roi;
}

vector<Mat> get_img_list(string data_dir)
{
	vector<string> file_list;
	list_dir(data_dir, file_list);
	int file_num = (int)(file_list.size());

	//read all img in RAM for quick access
	vector<Mat> img_list(file_num);
	cout << "start reading: " << endl;
	for (int i = 0; i < file_num; ++i)
	{
		string file_name = data_dir + "\\" + file_list[i];
		Mat img = imread(file_name);
		if (img.empty())
		{
			cerr << "Can not load image " << file_name << endl;
			system("pause");
			return img_list;
		}
		img_list[i] = img;

		//print progress
		cout << (100 * i / file_num) << "%";
		cout << "\b\b\b\b";
	}

	return img_list;
}

void generate_neg_data(string data_dir, string file_name, Size data_shape, int sample_size)
{
	vector<string> file_list;
	list_dir(data_dir, file_list);
	int file_num = (int)(file_list.size());
	int feature_dim = data_shape.height * data_shape.width * 10 / 16; //5120
	Mat X_neg(sample_size, feature_dim, CV_32F);

	//read all img in RAM for quick access
	vector<Mat> img_list(file_num);
	cout << "start reading: " << endl;
	for (int i = 0; i < file_num; ++i)
	{
		string file_name = data_dir + "\\" + file_list[i];
		Mat img = imread(file_name);
		if (img.empty())
		{
			cerr << "Can not load image " << file_name << endl;
			system("pause");
			return;
		}
		img_list[i] = img;

		//print progress
		cout << (100 * i / file_num) << "%";
		cout << "\b\b\b\b";
	}

	//random crop negative samples
	cout << "start croping: " << endl;
	for (int i = 0; i < sample_size; ++i)
	{
		int rand_index = std::abs(get_random()) % file_num;
		Mat roi = random_crop(img_list[rand_index]);
		Mat acf = bgr2acf(roi);
		acf = acf.reshape(1, 1);
		acf.copyTo(X_neg.row(i));

		//print progress
		cout << (100 * i / sample_size) << "%";
		cout << "\b\b\b\b";
	}

	cout << X_neg.size << endl;
	double max_v, min_v;
	minMaxLoc(X_neg, &min_v, &max_v);
	cout << "min: " << min_v << " max: " << max_v << endl;

	matwrite(file_name, X_neg);
}

void generate_hard_neg(string data_dir, string file_name, string clf_path, Size data_shape, int sample_size, int batch_size)
{
	Ptr<ml::Boost> clf = ml::Boost::load(clf_path);

	vector<Mat> img_list = get_img_list(data_dir);
	int file_num = (int)(img_list.size());
	int feature_dim = data_shape.height * data_shape.width * 10 / 16; //5120
	Mat X_neg(sample_size, feature_dim, CV_32F);

	int index = 0;
	while (index < sample_size)
	{
		//batch predict for speed acceleration
		Mat features(batch_size, feature_dim, CV_32F);
		for (int i = 0; i < batch_size; ++i)
		{
			int rand_index = std::abs(get_random()) % file_num;
			Mat roi = random_crop(img_list[rand_index]);
			Mat acf = bgr2acf(roi);
			acf = acf.reshape(1, 1);
			acf.copyTo(features.row(i));
		}
		Mat pred(batch_size, 1, CV_32F);
		clf->predict(features, pred, 0);
		const float* ppred = pred.ptr<float>(0);
		for (int i = 0; i < batch_size; ++i)
		{
			if (ppred[i])
			{
				features.row(i).copyTo(X_neg.row(index++));
			}
		}

		//print progress
		cout << (100 * index / sample_size) << "%";
		cout << "\b\b\b\b";
	}

	cout << X_neg.size << endl;
	double max_v, min_v;
	minMaxLoc(X_neg, &min_v, &max_v);
	cout << "min: " << min_v << " max: " << max_v << endl;

	matwrite(file_name, X_neg);
}

void train(string file_name)
{
	//prepare training data
	ifstream infile1("X_pos.dat");
	if (!infile1.good())
	{
		cerr << "Don't have X_pos.dat!" << endl;
		return;
	}
	infile1.close();
	ifstream infile2("X_neg.dat");
	if (!infile2.good())
	{
		cerr << "Don't have X_neg.dat!" << endl;
		return;
	}
	infile2.close();

	Mat X_pos = matread("X_pos.dat");
	Mat X_neg = matread("X_neg.dat");
	Mat y_pos(X_pos.rows, 1, CV_32S, Scalar(1));
	Mat y_neg(X_neg.rows, 1, CV_32S, Scalar(0));

    //concatenate data
	Mat X;
	X.push_back(X_pos);
	X.push_back(X_neg);

	Mat y;
	y.push_back(y_pos);
	y.push_back(y_neg);

	//hard negative
	Mat X_hard_neg = matread("X_hard_neg.dat");
	Mat y_hard_neg(X_hard_neg.rows, 1, CV_32S, Scalar(0));
	X.push_back(X_hard_neg);
	y.push_back(y_hard_neg);

	//train
	Ptr<ml::Boost> clf = ml::Boost::create();
	clf->setBoostType(ml::Boost::REAL);
	clf->setMaxDepth(2);
	clf->setWeakCount(1024);
	clf->setWeightTrimRate(0.95);
	clf->setUseSurrogates(false);
	clf->setPriors(Mat());

	clf->train(X, ml::ROW_SAMPLE, y);
	clf->save(file_name);
}

void evaluate(string clf_path)
{
	Mat X_pos = matread("X_pos_test.dat");
	Mat X_neg = matread("X_neg_test.dat");
	Mat y_pos(X_pos.rows, 1, CV_32S, Scalar(1));
	Mat y_neg(X_neg.rows, 1, CV_32S, Scalar(0));

	//concatenate data
	Mat X;
	X.push_back(X_pos);
	X.push_back(X_neg);

	Mat y;
	y.push_back(y_pos);
	y.push_back(y_neg);

	Ptr<ml::Boost> clf = ml::Boost::load(clf_path);
	Mat pred(y.size(), CV_32F);

	clock_t start = clock();
	float test_error = clf->predict(X, pred, 0);
	clock_t end = clock();
	cout << "prediction time: " << (float)(end - start) / y.rows << "ms" << endl;
	//cout << pred.type() << y.type() << endl;
	y.convertTo(y, CV_32F);
	Mat mask = (pred == y)/255;
	float total = (float)sum(mask)[0];
	float accuracy = 100.0f * total / y.rows;
	cout << "accuracy: " << accuracy << endl;
}