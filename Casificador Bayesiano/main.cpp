#include <iostream>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

void MatrizConfusion(Mat* result);

void EscribirClase(int i);

void ClasificarImagenes(Ptr<NormalBayesClassifier> Bayes, Ptr<TrainData>& test_data, Mat* result, Mat* prob, string* csv_loc);


int main(int argv, char** argc)
{

	// entrenamiento del clasificador bayesiano
	Ptr<TrainData> datos_entr = TrainData::loadFromCSV("datos_entrenamiento.csv", 1, 0, 1);
	Ptr<NormalBayesClassifier> Bayes = NormalBayesClassifier::create();
	Bayes->train(datos_entr);
	cout << "Entrenamiento del clasificador bayesiano completado\n\n" << endl;
	Bayes->save("clasificador_bayesiano.xml");


	// clasificamos las imagenes de test, lo hacemos en 8 tandas para saber cual es la respuesta correcta y poder sacar conclusiones
	// la matriz result almacena el resultado de la clasificacion. la matriz prob almacena la probabilidad de cada prediccion
	cout << "Clasificando imagenes de test...\n\n\n";
	Ptr<TrainData> test_data;
	Mat result[8], prob[8], test;

	string csv_loc[8] = { "tests\\ajo.csv", "tests\\lima.csv", "tests\\limon.csv", "tests\\manzana.csv", "tests\\nabo.csv", "tests\\pimiento.csv", "tests\\pina.csv", "tests\\tomate.csv" };

	ClasificarImagenes(Bayes, test_data, result, prob, csv_loc);

	MatrizConfusion(result);

	return 0;
}


void ClasificarImagenes(Ptr<NormalBayesClassifier> Bayes, Ptr<TrainData>& test_data, Mat* result, Mat* prob, string* csv_loc) {
	Mat test;

	for (int i = 0; i < 8; i++) {
		test_data = TrainData::loadFromCSV(csv_loc[i], 1, 0, 1);
		test = test_data->getTrainSamples();
		Bayes->predictProb(test, result[i], prob[i]);
	}
}

// tabulamos los resultados en forma de matriz confusion y sus estadisticas
void MatrizConfusion(Mat* result) {

	int m[8][8] = { 0 };
	int aciertos = 0, pos;
	float tp[8], fp[8] = { 0 }, prec[8] = { 0 }, recall[8], f1[8];

	cout << "\n******************************* MATRIZ DE CONFUSION ***********************************\n" << endl;
	cout << "Clasificado como  -->\t  Ajo\t Lima\tLimon\tManzana\t Nabo\tPiment\t Pina\tTomate\t" << endl;
	cout << "=======================================================================================";

	// primer bucle donde se calcularan y escribiran los elementos de la matriz de confusion
	for (int i = 0; i < 8; i++) {

		cout << endl;
		EscribirClase(i);
		cout << "\t|";

		for (int j = 0; j < result[i].rows; j++) {
			int c = result[i].at<int>(j, 0);
			m[i][c - 1]++;

		}
		for (int j = 0; j < 8; j++) {
			cout << "\t  " << m[i][j];
			if (i == j)
				aciertos += m[i][j];
		}
	}

	cout << "\n\nLa precision total del clasificador ha sido del " << 100 * aciertos / 80.f << "%" << endl;
	cout << m[1][6];
	cout << "\n_______________________________________________________________________________________" << endl;
	cout << "\nCalculo de los ratios de exito y error en la clasificacion:\n" << endl;
	cout << "\t\tRatio TP\tRatio FP\tPrecision\t Recall\t\tF1-score" << endl;


	// segundo bucle donde se calcularan y escribiran las estadisticas del test del clasificador
	for (int i = 0; i < 8; i++) {

		EscribirClase(i);

		pos = 0;
		for (int j = 0; j < 8; j++)
			pos += m[j][i];

		tp[i] = m[i][i] / 10.f;
		fp[i] = (10 - m[i][i]) / 70.f;
		recall[i] = tp[i];
		prec[i] = m[i][i] / (float)pos;
		f1[i] = 2 * (recall[i] * prec[i]) / (recall[i] + prec[i]);

		cout << setprecision(2) << "\t  " << tp[i] << "\t\t  " << fp[i] << "\t\t   " << prec[i] << "\t\t   " << recall[i] << "\t\t   " << f1[i] << endl;
	}
}


void EscribirClase(int i) {
	if (i == 0)
		cout << "Ajo\t";
	if (i == 1)
		cout << "Lima\t";
	if (i == 2)
		cout << "Limon\t";
	if (i == 3)
		cout << "Manzana\t";
	if (i == 4)
		cout << "Nabo\t";
	if (i == 5)
		cout << "Pimiento";
	if (i == 6)
		cout << "Pina\t";
	if (i == 7)
		cout << "Tomate\t";
}