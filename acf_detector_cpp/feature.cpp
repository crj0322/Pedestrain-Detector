#include "acf.h"

using namespace cv;
using namespace std;

Mat bgr2acf(Mat& img, int pool_size)
{
	//img info
	int channels = img.channels();
	int rows = img.rows;
	int cols = img.cols;

	//smooth
	Mat smooth_kernel = (Mat_<float>(1, 3) << 0.25, 0.5, 0.25);
	Mat img_smooth(rows, cols, CV_32FC3);
	img.convertTo(img, CV_32F);
	filter2D(img, img_smooth, -1, smooth_kernel);

	//calc gradient
	Mat img_dx(rows, cols, CV_32FC3);
	Mat img_dy(rows, cols, CV_32FC3);
	Sobel(img_smooth, img_dx, -1, 1, 0, 1);
	Sobel(img_smooth, img_dy, -1, 0, 1, 1);
	Mat img_mag(rows, cols, CV_32FC3);
	magnitude(img_dx, img_dy, img_mag);

	//get max gradient along channels
	Mat max_dx(rows, cols, CV_32FC1);
	Mat max_dy(rows, cols, CV_32FC1);
	Mat max_mag(rows, cols, CV_32FC1);

	//normally created matrix is continuous
	const float* pdx = img_dx.ptr<float>(0);
	const float* pdy = img_dy.ptr<float>(0);
	const float* pmag = img_mag.ptr<float>(0);
	float* pmax_dx = max_dx.ptr<float>(0);
	float* pmax_dy = max_dy.ptr<float>(0);
	float* pmax_mag = max_mag.ptr<float>(0);
	for (int i = 0, j = 0; i < rows * cols; ++i, j += channels)
	{
		float max_value = 0;
		for (int k = 0; k < channels; ++k)
		{
			if (pmag[j + k] >= max_value)
			{
				pmax_dx[i] = pdx[j + k];
				pmax_dy[i] = pdy[j + k];
				pmax_mag[i] = pmag[j + k];
				max_value = pmag[j + k];
			}
		}
	}

	//normalized gradient magnitude
	Mat exp_mag(rows, cols, CV_32FC1);
	boxFilter(max_mag, exp_mag, -1, Size(11, 11)); //to be replaced by triangle filter
	add(exp_mag, 0.005, exp_mag);
	divide(max_mag, exp_mag, max_mag);

	//gradient angle
	Mat grad_angle(rows, cols, CV_32FC1);
	phase(max_dx, max_dy, grad_angle);

	//hog channels of 6 bins
	Mat hog(rows, cols, CV_32FC(6), Scalar(0));
	float angle_start[6], angle_end[6];
	for (int i = 0; i < 6; ++i)
	{
		angle_start[i] = i * PI / 3;
		angle_end[i] = angle_start[i] + PI / 3;
	}

	const float* pgrad_angel = grad_angle.ptr<float>(0);
	float* phog_channels = hog.ptr<float>(0);
	for (int i = 0, j = 0; i < rows * cols; ++i, j += 6)
	{
		for (int k = 0; k < 6; ++k)
		{
			if (pgrad_angel[i] >= angle_start[k] && pgrad_angel[i] < angle_end[k])
			{
				phog_channels[j + k] = pmax_mag[i];
			}
		}
	}

	vector<Mat> hog_channels;
	split(hog, hog_channels);

	//bgr to luv
	//divide(img_smooth, 255.f, img_smooth);
	Mat img_luv(rows, cols, CV_32FC3);
	bgr2luv(img_smooth.ptr<float>(0), img_luv.ptr<float>(0), rows*cols, 1.f/255.f);
	vector<Mat> luv_channels;
	split(img_luv, luv_channels);

	//concatenate channels
	vector<Mat> feature_channels;
	feature_channels.push_back(max_mag);
	for (int i = 0; i < hog_channels.size(); ++i)
	{
		feature_channels.push_back(hog_channels[i]);
	}
	for (int i = 0; i < luv_channels.size(); ++i)
	{
		feature_channels.push_back(luv_channels[i]);
	}

	//sum pooling
	int out_rows = rows / pool_size;
	int out_cols = cols / pool_size;
	int out_chns = int(feature_channels.size());
	Mat acf(out_rows, out_cols, CV_32FC(10), Scalar(0));
	float* pacf = acf.ptr<float>(0);
	for (int k = 0; k < out_chns; ++k)
	{
		const float* pfeature = feature_channels[k].ptr<float>(0);
		for (int i = 0; i < out_rows; ++i)
		{
			for (int j = 0; j < out_cols; ++j)
			{
				float pool_sum = 0;
				for (int ii = 0; ii < pool_size; ++ii)
				{
					for (int jj = 0; jj < pool_size; ++jj)
					{
						pool_sum += pfeature[(pool_size*i + ii)*cols + pool_size * j + jj];
					}
				}
				pacf[i*out_cols*out_chns + j * out_chns + k] = pool_sum;
			}
		}
	}

	//smooth
	filter2D(acf, acf, -1, smooth_kernel);

	return acf;
}

// Constants for rgb2luv conversion and lookup table for y-> l conversion
template<class oT> oT* rgb2luv_setup(oT z, oT *mr, oT *mg, oT *mb,
	oT &minu, oT &minv, oT &un, oT &vn)
{
	// set constants for conversion
	const oT y0 = (oT)((6.0 / 29)*(6.0 / 29)*(6.0 / 29));
	const oT a = (oT)((29.0 / 3)*(29.0 / 3)*(29.0 / 3));
	un = (oT) 0.197833; vn = (oT) 0.468331;
	mr[0] = (oT) 0.430574*z; mr[1] = (oT) 0.222015*z; mr[2] = (oT) 0.020183*z;
	mg[0] = (oT) 0.341550*z; mg[1] = (oT) 0.706655*z; mg[2] = (oT) 0.129553*z;
	mb[0] = (oT) 0.178325*z; mb[1] = (oT) 0.071330*z; mb[2] = (oT) 0.939180*z;
	oT maxi = (oT) 1.0 / 270; minu = -88 * maxi; minv = -134 * maxi;
	// build (padded) lookup table for y->l conversion assuming y in [0,1]
	static oT lTable[1064]; static bool lInit = false;
	if (lInit) return lTable; oT y, l;
	for (int i = 0; i < 1025; i++) {
		y = (oT)(i / 1024.0);
		l = y > y0 ? 116 * (oT)pow((double)y, 1.0 / 3.0) - 16 : y * a;
		lTable[i] = l * maxi;
	}
	for (int i = 1025; i < 1064; i++) lTable[i] = lTable[i - 1];
	lInit = true; return lTable;
}

// Convert from rgb to luv
template<class iT, class oT> void bgr2luv(iT *I, oT *J, int n, oT nrm) {
	oT minu, minv, un, vn, mr[3], mg[3], mb[3];
	oT *lTable = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
	oT *L = J, *U = L + 1, *V = U + 1; iT *B = I, *G = B + 1, *R = G + 1;
	for (int i = 0; i < n; i++) {
		oT r, g, b, x, y, z, l;
		r = (oT)*R; g = (oT)*G; b = (oT)*B;
		x = mr[0] * r + mg[0] * g + mb[0] * b;
		y = mr[1] * r + mg[1] * g + mb[1] * b;
		z = mr[2] * r + mg[2] * g + mb[2] * b;
		l = lTable[(int)(y * 1024)];
		*L = l; z = 1 / (x + 15 * y + 3 * z + (oT)1e-35);
		*U = l * (13 * 4 * x*z - 13 * un) - minu;
		*V = l * (13 * 9 * y*z - 13 * vn) - minv;
		L += 3; U += 3; V += 3;
		R += 3; G += 3; B += 3;
	}
}