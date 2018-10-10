#include "acf.h"

using namespace std;
using namespace cv;

Mat detect_withoutapprox(Mat img, Ptr<ml::Boost> clf, Size window_wh, int pool_size, int stride)
{
	//row as (x, y, w, h, confidence)
	Mat bbox_list;

	//batch predict for speed acceleration
	Mat batch_samples;

	//row as (x, y, pyramid_count)
	Mat box_info;

	int pyramid_count = 0;
	Size slide_size = Size(window_wh.width / pool_size, window_wh.height / pool_size);
	while (true)
	{
		if (pyramid_count > 0)
		{
			pyrDown(img, img);
		}
		if (img.rows < window_wh.height || img.cols < window_wh.width)
		{
			break;
		}
		Mat feature_map = bgr2acf(img);

		//slide
		for (int y = 0; y < feature_map.rows; y += stride)
		{
			for (int x = 0; x < feature_map.cols; x += stride)
			{
				if (x + slide_size.width > feature_map.cols || y + slide_size.height > feature_map.rows)
					continue;
				Mat roi = feature_map(Rect(x, y, slide_size.width, slide_size.height));
				roi = roi.clone().reshape(1, 1);
				batch_samples.push_back(roi);
				Mat info = (Mat_<float>(1, 3) << x, y, pyramid_count);
				box_info.push_back(info);
			}
		}

		pyramid_count += 1;
	}

	//batch predict
	Mat confidence;
	clf->predict(batch_samples, confidence, ml::Boost::PREDICT_SUM);

	//calc bbox
	const float* ppred = confidence.ptr<float>(0);
	const float* pinfo = box_info.ptr<float>(0);
	for (int i = 0; i < confidence.rows; ++i)
	{
		if (ppred[i] > 0)
		{
			float x = pinfo[3 * i];
			float y = pinfo[3 * i + 1];
			float scale = pinfo[3 * i + 2];
			scale = (float)(std::pow(2, scale));

			x = pool_size * x * scale;
			y = pool_size * y * scale;
			float w = window_wh.width * scale;
			float h = window_wh.height * scale;
			Mat bbox = (Mat_<float>(1, 5) << x, y, w, h, ppred[i]);
			bbox_list.push_back(bbox);
		}
	}

	return bbox_list;
}

Mat detect(Mat img, Ptr<ml::Boost> clf, Size window_wh, int pool_size, int stride)
{
	//row as (x, y, w, h, confidence)
	Mat bbox_list;

	//batch predict for speed acceleration
	Mat batch_samples;

	//row as (x, y, pyramid_count)
	Mat box_info;

	Size slide_size = Size(window_wh.width / pool_size, window_wh.height / pool_size);
	for (int pyramid_count = 0; pyramid_count < 3; ++pyramid_count)
	{
		if (pyramid_count > 0)
		{
			pyrDown(img, img);
		}
		if (img.rows < window_wh.height || img.cols < window_wh.width)
		{
			break;
		}
		Mat feature_map = bgr2acf(img);
		float apprx_scale[4] = { 1.125f, 1.f, 0.875, 0.75 };
		for (int i = 0; i < 4; ++i)
		{
			if (pyramid_count == 0 && i == 0)
				continue;
			if (pyramid_count == 2 && i > 1)
				break;

			Mat scaled_map = feature_map.clone();
			Size scaled_size = Size((int)(apprx_scale[i] * feature_map.cols), (int)(apprx_scale[i] * feature_map.rows));
			if (i != 1)
			{
				resize(scaled_map, scaled_map, scaled_size);

				//for normalized gradient (first 7 channels), lambda=0.101
				float scale_factor = (float)(std::pow(apprx_scale[i], -0.101));
				vector<Mat> channels;
				cv::split(scaled_map, channels);
				for (int j = 0; j < 7; j++)
				{
					channels[j] *= scale_factor;
				}
				cv::merge(channels, scaled_map);
			}

			//slide
			for (int y = 0; y < scaled_map.rows; y += stride)
			{
				for (int x = 0; x < scaled_map.cols; x += stride)
				{
					if (x + slide_size.width > scaled_map.cols || y + slide_size.height > scaled_map.rows)
						continue;
					Mat roi = scaled_map(Rect(x, y, slide_size.width, slide_size.height));
					roi = roi.clone().reshape(1, 1);
					batch_samples.push_back(roi);
					float true_scale = (float)(std::pow(2, pyramid_count) * apprx_scale[i]);
					Mat info = (Mat_<float>(1, 3) << x, y, true_scale);
					box_info.push_back(info);
				}
			}
		}
	}

	//batch predict
	Mat confidence;
	clf->predict(batch_samples, confidence, ml::Boost::PREDICT_SUM);

	//calc bbox
	const float* ppred = confidence.ptr<float>(0);
	const float* pinfo = box_info.ptr<float>(0);
	for (int i = 0; i < confidence.rows; ++i)
	{
		if (ppred[i] > 0)
		{
			float x = pinfo[3 * i];
			float y = pinfo[3 * i + 1];
			float scale = pinfo[3 * i + 2];

			x = pool_size * x * scale;
			y = pool_size * y * scale;
			float w = window_wh.width * scale;
			float h = window_wh.height * scale;
			Mat bbox = (Mat_<float>(1, 5) << x, y, w, h, ppred[i]);
			bbox_list.push_back(bbox);
		}
	}

	return bbox_list;
}

Mat slice_row(Mat input, vector<int> idx)
{
	int rows = (int)idx.size();
	Mat output(rows, input.cols, input.type());
	for (int i = 0; i < idx.size(); ++i)
	{
		input.row(idx[i]).copyTo(output.row(i));
	}

	return output;
}

Mat non_max_suppression(Mat bbox_list, float iou_threshold)
{
	//sort with confidence
	Mat idxs;
	sortIdx(bbox_list.col(4), idxs, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
	vector<int> idx_list(idxs.begin<int>(), idxs.end<int>());

	//bbox info
	Mat x1 = bbox_list.col(0);
	Mat y1 = bbox_list.col(1);
	Mat w = bbox_list.col(2);
	Mat h = bbox_list.col(3);
	Mat x2 = x1 + w;
	Mat y2 = y1 + h;
	Mat area = (w + 1).mul(h + 1);

	//initialize the list of picked indexes
	vector<int> pick;

	//keep looping while some indexes still remain in the indexes list
	while (idx_list.size() > 0)
	{
		//grab the last index in the indexes list and add the index value
		//to the list of picked indexes
		int last = (int)(idx_list.size() - 1);
		int i = idx_list[last];
		pick.push_back(i);
		idx_list.pop_back();
		if (idx_list.size() == 0)
			break;

		//find the largest(x, y) coordinates for the start of the bounding box
		//and the smallest(x, y) coordinates for the end of the bounding box
		Mat xx1 = slice_row(x1, idx_list);
		Mat yy1 = slice_row(y1, idx_list);
		Mat xx2 = slice_row(x2, idx_list);
		Mat yy2 = slice_row(y2, idx_list);
		xx1 = cv::max(xx1, x1.at<float>(i, 0));
		yy1 = cv::max(yy1, y1.at<float>(i, 0));
		xx2 = cv::min(xx2, x2.at<float>(i, 0));
		yy2 = cv::min(yy2, y2.at<float>(i, 0));

		//compute the width and height of the bounding box
		Mat ww = cv::max(0, xx2 - xx1 + 1);
		Mat hh = cv::max(0, yy2 - yy1 + 1);

		//compute the ratio of overlap
		float area_pick = w.at<float>(i, 0) * h.at<float>(i, 0);
		Mat overlap;
		Mat intersection = ww.mul(hh);
		cv::divide(intersection, slice_row(area, idx_list) + area_pick - intersection, overlap);
		//cout << overlap << endl;

		//delete all indexes from the index list that have overlap greater
		//than the provided overlap threshold
		for (vector<int>::iterator it = idx_list.begin(); it != idx_list.end();)
		{
			int index = (int)(it - idx_list.begin());
			if (overlap.at<float>(index, 0) >= iou_threshold)
			{
				it = idx_list.erase(it);
			}
			else
			{
				++it;
			}
		}
	}

	//return only the bounding boxes that were picked
	return slice_row(bbox_list, pick);
}

void draw_bbox(Mat& img, Mat bbox_list)
{
	for (int i = 0; i < bbox_list.rows; i++)
	{
		const float* p_bbox = bbox_list.ptr<float>(i);
		int x = (int)p_bbox[0];
		int y = (int)p_bbox[1];
		int w = (int)p_bbox[2];
		int h = (int)p_bbox[3];
		float confidence = p_bbox[4];
		rectangle(img, Rect(x, y, w, h), Scalar(255, 0, 0), 2);
		ostringstream ss;
		ss << setiosflags(ios::fixed) << setprecision(3) << confidence;
		std::string text(ss.str());
		putText(img, text, Point(x, y), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
	}
}