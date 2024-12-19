#include "guidedfilter.h"
#include <stdio.h>

using namespace cv;
using namespace std;

#define TMIN	(25)
#define DEFAULT_WINDOW_SIZE		(60)
#define DEFAULT_WEIGHT			(85)

void createDarkChannel(Mat in, Mat out, int windowSize);
void getAtmLight(Mat in, int *atmLight);
void transmissionEstimate(Mat in, Mat out, int *atmLight, int weight);
void getRadiance(Mat in, Mat transmission, Mat out, int *atmLight);

int main(int argc, const char* argv[])
{
	int atmLight[3];
	int weight = DEFAULT_WEIGHT;
	int window_size = DEFAULT_WINDOW_SIZE;

	if (argc < 2) {
		printf("%s dehazes image.\n", argv[0]);
		printf("%s <input image> [<dehaze weight(1-99)> <window size(1-255)> <dehazed filename>]\n", argv[0]);
		return -1;
	}

	if (argc > 2) {
		weight = atoi(argv[2]);
	}
	printf("weight:%d\n", weight);

	if (argc > 3) {
		window_size = atoi(argv[3]);
	}
	if (window_size < 1) window_size = 1;
	if (window_size > 256) window_size = 255;
	printf("window_size:%d\n", window_size);

	// ディヘイズする画像をファイルから読み込む
	Mat in = imread(argv[1], IMREAD_COLOR);

	if (in.empty()) {
		printf("Failed to load image (%s).\n", argv[1]);
		return -1;
	}

	// ウィンドウ内の最小値でダークチャネルを生成する
	Mat darkCh(in.rows, in.cols, in.type());
	createDarkChannel(in, darkCh, window_size);
	if (darkCh.empty()) {
		printf("Failed to createDarkChannel\n");
		return -1;
	}

	getAtmLight(darkCh, atmLight);

	// Guided Filterでダークチャネルを精緻化する
	Mat darkCh2(in.rows, in.cols, in.type());
	int r = 60;
	double eps = 1e-6;
	eps *= 255 * 255;   // Because the intensity range of our images is [0, 255]

	darkCh2 = guidedFilter(in, darkCh, r, eps);
	if (darkCh2.empty()) {
		printf("Failed to execute guided filter.\n");
		return -1;
	}

	// 光の透過率を求める
	Mat transmission(in.rows, in.cols, in.type());
	transmissionEstimate(darkCh2, transmission, atmLight, weight);
	if (transmission.empty()) {
		printf("Failed to execute transmissionEstimate.\n");
		return -1;
	}

	// 透過率に応じてコントラストを調整する
	Mat out(in.rows, in.cols, in.type());
	getRadiance(in, transmission, out, atmLight);
	if (out.empty()) {
		printf("Failed to execute getRadiance.\n");
		return -1;
	}

	// 入出力画像を表示する
	namedWindow("original", WINDOW_AUTOSIZE);
	imshow("original", in);

	if (!out.empty()) {
		namedWindow("dehazed", WINDOW_AUTOSIZE);
		imshow("dehazed", out);
	}

	waitKey(0);

	// ディヘイズ画像を保存する
	if (argc > 4) {
		imwrite(argv[4], out);
	}
	else {
		imwrite("dehazed.png", out);
	}
	return 0;
}

void createDarkChannel(Mat in, Mat out, int windowSize)
{
	if ((in.empty()) || (in.rows != out.rows) || (in.cols != out.cols) || (in.type() != out.type())) {
		printf("Invalid parameter\n");
		return;
	}

	Mat tmp1(in.rows, in.cols, in.type());
	Mat tmp2(in.rows, in.cols, in.type());
	// １パス目：水平方向にウィンドウ内の最小値を求める in-->tmp1
	for (int y = 0; y < in.rows; y++) {
		for (int x = 0; x < in.cols; x++) {
			int bmin = 255;
			int gmin = 255;
			int rmin = 255;
			for (int i = 0; i < windowSize; i++) {
				int xx = x + i - windowSize / 2;
				if (xx < 0) continue;
				if (xx >= in.cols) continue;

				Vec3b bgr = in.at<Vec3b>(y, xx);
				int b = bgr[0];
				if (b < bmin) bmin = b;
				int g = bgr[1];
				if (g < gmin) gmin = g;
				int r = bgr[2];
				if (r < rmin) rmin = r;
			}
			tmp1.at<Vec3b>(y, x) = Vec3b(bmin, gmin, rmin);
		}
	}

	// ２パス目：垂直方向にウィンドウ内の最小値を求める tmp1-->tmp2
	for (int x = 0; x < tmp1.cols; x++) {
		for (int y = 0; y < tmp1.rows; y++) {
			int bmin = 255;
			int gmin = 255;
			int rmin = 255;
			for (int i = 0; i < windowSize; i++) {
				int yy = y + i - windowSize / 2;
				if (yy < 0) continue;
				if (yy >= tmp1.rows) continue;

				Vec3b bgr = tmp1.at<Vec3b>(yy, x);
				int b = bgr[0];
				if (b < bmin) bmin = b;
				int g = bgr[1];
				if (g < gmin) gmin = g;
				int r = bgr[2];
				if (r < rmin) rmin = r;
			}
			tmp2.at<Vec3b>(y, x) = Vec3b(bmin, gmin, rmin);
		}
	}

	// ３パス目：水平方向にウィンドウ内の最大値を求める tmp2-->tmp1
	for (int y = 0; y < tmp2.rows; y++) {
		for (int x = 0; x < tmp2.cols; x++) {
			int bmax = 0;
			int gmax = 0;
			int rmax = 0;
			for (int i = 0; i < windowSize; i++) {
				int xx = x + i - windowSize / 2;
				if (xx < 0) continue;
				if (xx >= tmp2.cols) continue;

				Vec3b bgr = tmp2.at<Vec3b>(y, xx);
				int b = bgr[0];
				if (b > bmax) bmax = b;
				int g = bgr[1];
				if (g > gmax) gmax = g;
				int r = bgr[2];
				if (r > rmax) rmax = r;
			}
			tmp1.at<Vec3b>(y, x) = Vec3b(bmax, gmax, rmax);
		}
	}

	// ４パス目：垂直方向にウィンドウ内の最大値を求める tmp1-->out
	for (int x = 0; x < tmp1.cols; x++) {
		for (int y = 0; y < tmp1.rows; y++) {
			int bmax = 0;
			int gmax = 0;
			int rmax = 0;
			for (int i = 0; i < windowSize; i++) {
				int yy = y + i - windowSize / 2;
				if (yy < 0) continue;
				if (yy >= tmp1.rows) continue;

				Vec3b bgr = tmp1.at<Vec3b>(yy, x);
				int b = bgr[0];
				if (b > bmax) bmax = b;
				int g = bgr[1];
				if (g > gmax) gmax = g;
				int r = bgr[2];
				if (r > rmax) rmax = r;
			}
			out.at<Vec3b>(y, x) = Vec3b(bmax, gmax, rmax);
		}
	}
}

void getAtmLight(Mat in, int *atmLight)
{
	int histogram[3][256] = { 0 };
	int i;
	int th = in.rows * in.cols / 1000; // 0.1%

	for (int y = 0; y < in.rows; y++) {
		Vec3b *ptr = in.ptr<Vec3b>(y);
		for (int x = 0; x < in.cols; x++) {
			Vec3b bgr = ptr[x];

			++histogram[0][bgr[0]];
			++histogram[1][bgr[1]];
			++histogram[2][bgr[2]];
		}
	}

	for (int ch = 0; ch < 3; ch++) {
		int sum = 0;
		int atm = 0;

		for (i = 255; i >= 0; i--) {
			sum += histogram[ch][i];
			if (sum > th) break;
		}

		for (; i < 256; i++) {
			atm += i * histogram[ch][i];
		}

		atmLight[ch] = atm / sum;
	}
}

void transmissionEstimate(Mat in, Mat out, int *atmLight, int weight)
{
	if (in.empty() || (atmLight == NULL)) {
		printf("%s:Invalid parameter.\n", __FUNCTION__);
		return;
	}

	for (int y = 0; y < in.rows; y++) {
		Vec3b *src = in.ptr<Vec3b>(y);
		Vec3b *dst = out.ptr<Vec3b>(y);
		for (int x = 0; x < in.cols; x++) {
			Vec3b bgr = src[x];
			int b = (255 * 100 - weight * 255 * bgr[0] / atmLight[0]) / 100;
			int g = (255 * 100 - weight * 255 * bgr[1] / atmLight[1]) / 100;
			int r = (255 * 100 - weight * 255 * bgr[2] / atmLight[2]) / 100;

			dst[x] = Vec3b(b, g, r);
		}
	}
}

void getRadiance(Mat in, Mat transmission, Mat out, int *atmLight)
{
	if (in.empty() || transmission.empty() || (atmLight == NULL)) {
		printf("%s:Invalid parameter.\n", __FUNCTION__);
		return;
	}

	for (int y = 0; y < in.rows; y++) {
		Vec3b *src = in.ptr<Vec3b>(y);
		Vec3b *trans = transmission.ptr<Vec3b>(y);
		Vec3b *dst = out.ptr<Vec3b>(y);
		for (int x = 0; x < in.cols; x++) {
			Vec3b src_bgr = src[x];
			Vec3b trans_bgr = trans[x];
			if (trans_bgr[0] < TMIN) trans_bgr[0] = TMIN;
			if (trans_bgr[1] < TMIN) trans_bgr[1] = TMIN;
			if (trans_bgr[2] < TMIN) trans_bgr[2] = TMIN;

			int b = ((src_bgr[0] - atmLight[0]) * 255 + atmLight[0] * trans_bgr[0]) / trans_bgr[0];
			int g = ((src_bgr[1] - atmLight[1]) * 255 + atmLight[1] * trans_bgr[1]) / trans_bgr[1];
			int r = ((src_bgr[2] - atmLight[2]) * 255 + atmLight[2] * trans_bgr[2]) / trans_bgr[2];

			if (b > 255) b = 255; if (b < 0) b = 0;
			if (g > 255) g = 255; if (g < 0) g = 0;
			if (r > 255) r = 255; if (r < 0) r = 0;

			dst[x] = Vec3b(b, g, r);
		}
	}
}
