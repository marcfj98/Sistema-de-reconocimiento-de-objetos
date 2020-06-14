#include <iostream>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;


CascadeClassifier desc_cara, desc_ojos, desc_ojoi, desc_boca, desc_nariz;
const int FPS = 30; // fotogramas por segundo que tiene nuestra camara o el video

void DetectorFacciones(Mat &img);

int main(int argv, char** argc) {
	desc_cara.load("descriptores\\haarcascade_frontalface_default.xml"); // cargamos los descriptores de haar
	desc_ojos.load("descriptores\\haarcascade_eye.xml");
	desc_boca.load("descriptores\\haarcascade_smile.xml");
	string modo = "video"; // modo "video" si trabajamos con videos o camara, cualquier otro valor para trabajar con imagenes
	Mat img, bn;

	if (modo == "video") {
		//VideoCapture captura(0);
		VideoCapture captura("avg.mp4");
		while (waitKey(1000 / FPS) != 27) {
			captura.read(img);
			// si la resolución del vídeo es alta, el programa no es capaz de operar a tiempo real, es necesario reducirlo
			int nueva_res = 320; // las filas que tendrá la imagen reducida
			resize(img, img, Size(nueva_res * img.cols / img.rows, nueva_res));

			DetectorFacciones(img);

			imshow("Reconocimiento de caras", img);
		}
	}
	else {
		img = imread("gente.jpg");

		DetectorFacciones(img);

		imshow("Reconocimiento de caras", img);
		waitKey();
	}


	return 0;
}


void DetectorFacciones(Mat &img) {
	Mat bn;
	cvtColor(img, bn, COLOR_BGR2GRAY);
	vector<Rect> cara, ojos, boca, nariz;
	desc_cara.detectMultiScale(bn, cara, 1.1, 20);
	for (int i = 0; i < cara.size(); i++) {
		rectangle(img, Rect(cara[i].x, cara[i].y, cara[i].width, cara[i].height), Scalar(0, 0, 255), 3);
		putText(img, "Cara", Point(cara[i].x, cara[i].y - cara[i].height / 20), FONT_HERSHEY_SIMPLEX, cara[i].width / 120, Scalar(0, 0, 255), 2);

		Mat ROI_cara = bn(cara[i]);
		desc_ojos.detectMultiScale(ROI_cara, ojos, 1.1, 80);
		for (int j = 0; j < ojos.size(); j++) {
			Point centro(cara[i].x + ojos[j].x + ojos[j].width / 2, cara[i].y + ojos[j].y + ojos[j].height / 2);
			ellipse(img, centro, Size(ojos[j].width / 2, ojos[j].height / 2), 0, 0, 360, Scalar(255, 0, 0), 2);
			putText(img, "Ojo", Point(cara[i].x + ojos[j].x, cara[i].y + ojos[j].y - ojos[j].height / 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
		}

		desc_boca.detectMultiScale(ROI_cara, boca, 1.25, 35);
		for (int j = 0; j < boca.size(); j++) {
			Point centro(cara[i].x + boca[j].x + boca[j].width / 2, cara[i].y + boca[j].y + boca[j].height / 2);
			ellipse(img, centro, Size(boca[j].width / 2, boca[j].height / 2), 0, 0, 360, Scalar(0, 255, 0), 2);
			putText(img, "Boca", Point(cara[i].x + boca[j].x, cara[i].y + boca[j].y - boca[j].height / 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
		}

	}
}