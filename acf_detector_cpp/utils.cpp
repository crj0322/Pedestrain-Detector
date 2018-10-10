#include "acf.h"
#include <Windows.h>

using namespace std;
using namespace cv;

void list_dir(const string& name, vector<string>& v)
{
	string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			if (data.dwFileAttributes == FILE_ATTRIBUTE_ARCHIVE)
				v.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

void matwrite(const string& filename, const Mat& mat)
{
	ofstream fs(filename, fstream::binary);

	// Header
	int type = mat.type();
	int channels = mat.channels();
	fs.write((char*)&mat.rows, sizeof(int));    // rows
	fs.write((char*)&mat.cols, sizeof(int));    // cols
	fs.write((char*)&type, sizeof(int));        // type
	fs.write((char*)&channels, sizeof(int));    // channels

	// Data
	if (mat.isContinuous())
	{
		fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
	}
	else
	{
		int rowsz = CV_ELEM_SIZE(type) * mat.cols;
		for (int r = 0; r < mat.rows; ++r)
		{
			fs.write(mat.ptr<char>(r), rowsz);
		}
	}
}

Mat matread(const string& filename)
{
	ifstream fs(filename, fstream::binary);

	// Header
	int rows, cols, type, channels;
	fs.read((char*)&rows, sizeof(int));         // rows
	fs.read((char*)&cols, sizeof(int));         // cols
	fs.read((char*)&type, sizeof(int));         // type
	fs.read((char*)&channels, sizeof(int));     // channels

	// Data
	Mat mat(rows, cols, type);
	fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

	return mat;
}

int get_random()
{
	int rnum = 0;
#if defined _MSC_VER
#if defined _WIN32_WCE
	CeGenRandom(sizeof(int), (PBYTE)&rnum);
#else
	HCRYPTPROV hProvider = 0;
	const DWORD dwLength = sizeof(int);
	BYTE pbBuffer[dwLength] = {};
	DWORD result = ::CryptAcquireContext(&hProvider, 0, 0, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT | CRYPT_SILENT);
	assert(result);
	result = ::CryptGenRandom(hProvider, dwLength, pbBuffer);
	rnum = *(int*)pbBuffer;
	assert(result);
	::CryptReleaseContext(hProvider, 0);
#endif
#elif defined __GNUC__
	int fd = open("/dev/urandom", O_RDONLY);
	if (fd != -1) {
		(void)read(fd, (void *)&rnum, sizeof(int));
		(void)close(fd);
	}
#endif
	return rnum;
}
