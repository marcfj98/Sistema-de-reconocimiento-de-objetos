#include <fstream>
#include <iostream>
#include <cmath>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

typedef struct {
	string clase;
	double area;
	double perimetro;
	double cx;
	double cy;
	double compacidad;
	double longitud;
	double anchura;
	double excentricidad;
	double eje_p_inercia;
	int min_x;
	int min_y;
	int max_x;
	int max_y;
	int hue;
} caracteristicas;


void PreprocesadoSegmentacion(Mat img, Mat& hsv, Mat& bn);

void ExtraccionCaracteristicas(Mat bn, caracteristicas* car);	

// El programa tomará imagenes y extraerá sus características. Se crearán archivos datos.txt, datos.arff y datos.csv con los datos con un formato apropiado para analizar en Excel, en Weka o para importar a otro programa
// Es necesario un archivo imagenes.txt con los nombres de todas las imagenes a analizar.


/////////////////////////////////////////////////////////////////////////////////////////////
////////////// No se han añadido todas las imagenes para ahorrar espacio al entregar ////////
////////////// Solo se han dejado las imagenes de test de la lima para probar ///////////////
////////////// Pero se han dejado los resultados obtenidos //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////


int main(int argv, char** argc)
{
	Mat img, bn;
	ifstream imagenes;
	ofstream txt, weka, csv;
	string linea;
	string clase = "lima\\test"; // clase a la que pertenecen las imagenes que vamos a analizar. Se deberá cambiar para cada grupo de imagenes
	imagenes.open("imagenes\\" + clase +"\\imagenes.txt"); // .txt que contiene los nombres de las imagenes
	txt.open("imagenes\\" + clase + "\\datos.txt"); // .txt que contiene las caracteristicas extraidas
	weka.open("imagenes\\" + clase + "\\datos.arff");
	csv.open("imagenes\\" + clase + "\\datos.csv");
	caracteristicas car_img;

	//escribimos el encabezado de los archivos
	txt << "Clase\tArea\tPerimetro\tX minima\tY minima\tX maxima\tY maxima\tCDGx\tCDGy\tCompacidad\tExcentricidad\tEjePInercia\tLongitud\tAnchura\tHue" << endl;
	weka << "@relation Blobs\n@attribute Area real\n@attribute Perimetro real\n@attribute X_minima real\n@attribute Y_minima real\n@attribute X_maxima real\n@attribute Y_maxima real\n@attribute CDG_X real\n@attribute CDG_Y real\n"
		<< "@attribute Compacidad real\n@attribute Excentricidad real\n@attribute EjePInercia real\n@attribute Longitud real\n@attribute Anchura real\n@attribute Hue real\n@attribute Clase {a completar manualmente}\n@data" << endl;
	csv << "Clase, Area, Perimetro, X minima, Y minima, X maxima, Y maxima, CDGx, CDGy, Compacidad, Excentricidad, EjePInercia, Longitud, Anchura, Hue" << endl;


	//extraemos las caracteristicas de cada imagen y las escribimos en los archivos
	while (getline(imagenes, linea)) {

		car_img.clase = clase;

		img = imread("imagenes\\" + clase + "\\" + linea);

		ExtraccionCaracteristicas(img, &car_img); 

		txt << car_img.clase << "\t" << car_img.area << "\t" << car_img.perimetro << "\t" << car_img.min_x << "\t" << car_img.min_y << "\t" << car_img.max_x << "\t" << car_img.max_y << "\t"
			<< car_img.cx << "\t" << car_img.cy << "\t" << car_img.compacidad << "\t" << car_img.excentricidad << "\t" << car_img.eje_p_inercia << "\t" << car_img.longitud << "\t" << car_img.anchura << "\t" << car_img.hue << endl;
		weka << car_img.area << "\t" << car_img.perimetro << "\t" << car_img.min_x << "\t" << car_img.min_y << "\t" << car_img.max_x << "\t" << car_img.max_y << "\t" << car_img.cx
			<< "\t" << car_img.cy << "\t" << car_img.compacidad << "\t" << car_img.excentricidad << "\t" << car_img.eje_p_inercia << "\t" << car_img.longitud << "\t" << car_img.anchura << "\t" << car_img.hue << "\t" << car_img.clase << endl;
		csv << car_img.clase << ", " << car_img.area << ", " << car_img.perimetro << ", " << car_img.min_x << ", " << car_img.min_y << ", " << car_img.max_x << ", " << car_img.max_y << ", "
			<< car_img.cx << ", " << car_img.cy << ", " << car_img.compacidad << ", " << car_img.excentricidad << ", " << car_img.eje_p_inercia << ", " << car_img.longitud << ", " << car_img.anchura << ", " << car_img.hue << endl;
	}
	

	return 0;
}


// procesamiento y segmentacion de la imagen
void PreprocesadoSegmentacion(Mat img, Mat& hsv, Mat& bn) {


	Mat mascara = getStructuringElement(MORPH_ELLIPSE, Size(5, 5)); // mascara circular con la que se hara el opening

	// la umbralizacion se hara con el canal del valor de la imagen en hsv, ya que queremos eliminar los valores oscuros que conforman el fondo
	cvtColor(img, hsv, COLOR_BGR2HSV);
	inRange(hsv, Scalar(0, 0, 25), Scalar(255, 255, 255), bn);
	morphologyEx(bn, bn, MORPH_OPEN, mascara, Point(-1, -1), 2); // 2 openings para eliminar los bordes con salidas

	// calculo de los contornos para rellenarlos
	vector<vector<Point> > contornos;
	findContours(bn, contornos, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	double area_max = 0;
	int max_cnt = -1;
	for (int i = 0; i < contornos.size(); i++) {
		double area = contourArea(contornos[i]);
		if (area > area_max) {
			area_max = area;
			max_cnt = i;
		}
	}
	bn -= bn;
	drawContours(bn, contornos, max_cnt, Scalar(255, 255, 255), -1);

	imshow("segment", bn);
	waitKey(20);

}



// extraccion de las caracteristicas de la imagen, se almacenan en una estructura caracteristicas
void ExtraccionCaracteristicas(Mat img, caracteristicas* car) {

	Mat bn, hsv;
	PreprocesadoSegmentacion(img, hsv, bn);

	vector<vector<Point>> contorno;
	findContours(bn, contorno, RETR_EXTERNAL, CHAIN_APPROX_NONE); //contorno a partir del cual se calculan las caracteristicas

	car->area = contourArea(contorno[0]); //area

	car->perimetro = arcLength(contorno[0], true); //perimetro

	// centro de gravedad
	Moments M = moments(contorno[0], true);
	car->cx = M.m10 / M.m00;
	car->cy = M.m01 / M.m00;

	//eje principal de inercia
	car->eje_p_inercia = -0.5 * atan(2 * M.m11 / (M.mu20 - M.m02)) * 180 / 3.14159;

	// excentricidad
	car->excentricidad = (4 * pow(M.m11, 2) - pow(M.m20 - M.m02, 2)) / pow(M.m20 + M.m02, 2);

	// compacidad
	car->compacidad = 4 * 3.14159 * car->area / pow(car->perimetro, 2); // compacidad

	// x e y maximos y minimos
	Rect maxmin = boundingRect(contorno[0]);
	car->min_x = maxmin.x;
	car->min_y = maxmin.y;
	car->max_x = maxmin.x + maxmin.width;
	car->max_y = maxmin.y + maxmin.height;

	// longitud y anchura
	RotatedRect caja = minAreaRect(contorno[0]);
	Point2f puntos[4];
	caja.points(puntos);
	double d1, d2;
	d1 = abs(sqrt(pow(puntos[1].x - puntos[0].x, 2) + pow(puntos[1].y - puntos[0].y, 2)));
	d2 = abs(sqrt(pow(puntos[2].x - puntos[1].x, 2) + pow(puntos[2].y - puntos[1].y, 2)));
	if (d1 < d2)
		car->anchura = d1, car->longitud = d2;
	else
		car->anchura = d2, car->longitud = d1;

	// tono medio
	car->hue = (int)mean(hsv, bn)[0];
}
