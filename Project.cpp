#include <iostream>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <eigen3/Eigen/src/Core/BandMatrix.h>
#include <fstream>
#include <time.h>
 
using namespace Eigen;
using namespace std;




float mu = 1/81.45;
float eps = 0.0001;


float FU_base_1(float U1, float U2, float U3, float mu){
	float result;
	result = pow(pow(U2,2)+pow(U3,2)+pow(U1+mu-1,2),1.5);
	return result;
}

float FU_base_2(float U1, float U2, float U3, float mu){
	float result;
	result = pow(pow(U2,2)+pow(U3,2)+pow(U1+mu,2),1.5);
	return result;
}


MatrixXf F(MatrixXf UN) {
   MatrixXf result;
   int n = UN.cols();
   result = Eigen::Matrix<float,Dynamic,Dynamic>();
   result.resize(1,n);

   result(0,0) = UN(0,3);
   result(0,1) = UN(0,4);
   result(0,2) = UN(0,5);
 
   result(0,3) = 2.0 * UN(0,4) + UN(0,0) 
	- mu * (UN(0,0)+mu-1) / FU_base_1(UN(0,0),UN(0,1),UN(0,2),mu) 
	- (1-mu) * (UN(0,0)+mu) /  FU_base_2(UN(0,0),UN(0,1),UN(0,2),mu);
   result(0,4) = -2.0 * UN(0,3) + UN(0,1) 
	- mu * (UN(0,1)) / FU_base_1(UN(0,0),UN(0,1),UN(0,2),mu) 
	- (1-mu) * (UN(0,1)) /  FU_base_2(UN(0,0),UN(0,1),UN(0,2),mu);
	result(0,5) =  - mu * (UN(0,2)) /FU_base_1(UN(0,0),UN(0,1),UN(0,2),mu) 
	- (1-mu) * (UN(0,2)) /  FU_base_2(UN(0,0),UN(0,1),UN(0,2),mu);
   return result;

}

MatrixXf Ftest(MatrixXf UN) {
   MatrixXf result;
   result = UN.array().cos();
   return result;
}
// MatrixXf FixedPointIteration(MatrixXf C1, float C2, MatrixXf C3, float C4) {

// 	// X = C4 f(C2*X + C1) +C3
// 	MatrixXf result;
// 	int n = C1.cols();
// 	result = Eigen::Matrix<float,Dynamic,Dynamic>();
// 	result.resize(1,n);

	
// 	while(1){
// 		result = C4 * F(C2 * result + C1) -C3;
// 		if((result - C4 * F( C2 *result + C1) -C3).norm()<eps)break;
// 		//std::cout << result << std::endl;
// 	}
	
// 	return result;

// }



MatrixXf EulerMethod(MatrixXf U, float k, int N){
	for (int i = 0; i < N; i++){ 
		MatrixXf UN = U.row(i); // U^N
		MatrixXf y1 = F(UN);
		U.row(i+1) = U.row(i) +  k * y1;
		}
	return U;
}


MatrixXf ClassicRKInteration(MatrixXf UN, float k) {
	MatrixXf result;
	int n = UN.cols();
	//the function can be defined here *
	//float mu = 1/81.45;
	result = Eigen::Matrix<float,Dynamic,Dynamic>();
	result.resize(1,n);

	MatrixXf y1 = F(UN);
	MatrixXf y2 = F(UN + 0.5 * k * y1);
	MatrixXf y3 = F(UN + 0.5 * k * y2);
	MatrixXf y4 = F(UN + k * y3);

	result = UN + 1.0/6.0 * k * (y1 +2.0 * y2 + 2.0 * y3 + y4);
	return result;
}


MatrixXf ClassicRKMethod(MatrixXf U, float k, int N){
	for (int i = 0; i < N; i++) U.row(i+1) = ClassicRKInteration(U.row(i),k);
	return U;
}


MatrixXf AB1Method(MatrixXf U, float k, int N){
	U = EulerMethod(U,k,N);
	return U;
}

MatrixXf AB2Method(MatrixXf U, float k, int N){
	for (int i = 0; i < N; i++){
		if (i == 0){
			U.row(i+1) = ClassicRKInteration(U.row(i),k);
		}	 // Euler's method for U^1
		if (i <= N-2) U.row(i+2) = U.row(i+1) + 1.5 * k * F(U.row(i+1)) - 0.5 * k * F(U.row(i));
	}
	return U;
}

MatrixXf AB3Method(MatrixXf U, float k, int N){
	//----------Adams-Bashforth methods (p = 3)----------	
	for(int i = 0; i <= 1; i++){
		U.row(i+1) = ClassicRKInteration(U.row(i),k);
	}	 
	for (int i = 0; i <= N-3; i++){ 
		U.row(i+3) = U.row(i+2) + k* (
			23.0/12.0  * F(U.row(i+2)) 
			- 16.0/12.0  * F(U.row(i+1))	
			+ 5.0/12.0  * F(U.row(i)));
	}
	return U;
}

MatrixXf AB4Method(MatrixXf U, float k, int N){
	//----------Adams-Bashforth methods (p = 4)----------	
	for(int i = 0; i <= 2; i++){
		U.row(i+1) = ClassicRKInteration(U.row(i),k);
	}	 
	for (int i = 0; i <= N-4; i++){ 
		U.row(i+4) = U.row(i+3) + k* (
			55.0/24.0  * F(U.row(i+3)) 
			- 59.0/24.0  * F(U.row(i+2))	
			+ 37.0/24.0  * F(U.row(i+1))
			- 9.0/24.0 * F(U.row(i)));
	}
	return U;
}



MatrixXf BDF(MatrixXf UN, float k){

	int n = UN.cols();
	MatrixXf next;
	next = Eigen::Matrix<float,Dynamic,Dynamic>();
	next.resize(1,n);
	next = MatrixXf::Zero(1,n);
	while(1){
		next =  k * F(next) + UN;
		if((next - k * F(next) - UN).norm() < eps)
			break;
	}
	return next;

}

MatrixXf BDF1Method(MatrixXf U, float k, int N){

	for (int i = 0; i < N; i++){ 
		U.row(i+1) = BDF(U.row(i),k);
	}
	return U;
}

MatrixXf BDF2Method(MatrixXf U, float k, int N){

	for (int i = 0; i < 1; i++){ 
		U.row(i+1) = BDF(U.row(i),k);
	}
	for (int i = 0; i < N - 1; i++){ 
		U.row(i+2) = BDF(4.0/3.0 * U.row(i+1) -1.0/3.0 * U.row(i) ,2.0/3.0 * k);
	}
	return U;
}

MatrixXf BDF3Method(MatrixXf U, float k, int N){

	for (int i = 0; i < 2; i++){ 
		U.row(i+1) = BDF(U.row(i),k);
	}
	for (int i = 0; i < N - 2; i++){ 
		U.row(i+3) = BDF(18.0/11.0 * U.row(i+2) - 9.0/11.0 * U.row(i+1) + 2.0/11.0 * U.row(i) ,6.0/11.0 * k);
	}
	return U;
}

MatrixXf BDF4Method(MatrixXf U, float k, int N){

	for (int i = 0; i < 3; i++){ 
		U.row(i+1) = BDF(U.row(i),k);
	}
	for (int i = 0; i < N - 3; i++){ 
		U.row(i+4) = BDF(48.0/25.0 * U.row(i+3) - 36.0/25.0 * U.row(i+2) + 16.0/25.0 * U.row(i+1) - 3.0/25.0 * U.row(i) ,12.0/25.0 * k);
	}
	return U;
}




MatrixXf AM2Method(MatrixXf U, float k, int N){
	for(int i = 0; i < 1; i++){
		U.row(i+1) = ClassicRKInteration(U.row(i),k);
	}
	for (int i = 0; i < N; i++){
		U.row(i+1) = BDF(
			0.5 * k * F(U.row(i)) 
			+ U.row(i),
			0.5 * k);
	}
	return U;
}

MatrixXf AM3Method(MatrixXf U, float k, int N){
	for(int i = 0; i <= 1; i++){
		U.row(i+1) = ClassicRKInteration(U.row(i),k);
	}

	for (int i = 0; i <= N-2; i++){ 
		U.row(i+2) = BDF(
			U.row(i+1)
			+ 8.0/12.0 * k * F(U.row(i+1)) 
			- 1.0/12.0 * k * F(U.row(i)) 
			,5.0/12.0 * k);
	}
	return U;
}

MatrixXf AM4Method(MatrixXf U, float k, int N){
	for(int i = 0; i <= 2; i++){
		U.row(i+1) = ClassicRKInteration(U.row(i),k);
	}	 
	for (int i = 0; i <= N-3; i++){ 
		U.row(i+3) = BDF(
			U.row(i+2)
			+ 19.0/24.0 * k * F(U.row(i+2)) 
			- 5.0/24.0 * k * F(U.row(i+1)) 
			+ 1.0/24.0 * k * F(U.row(i)) 
			,9.0/24.0 * k);
	}
	return U;
}

MatrixXf AM5Method(MatrixXf U, float k, int N){
 	for(int i = 0; i <= 3; i++){
		U.row(i+1) = ClassicRKInteration(U.row(i),k);
	}	 
	for (int i = 0; i <= N-4; i++){ 
		U.row(i+4) = BDF(
			U.row(i+3)
			+ 646.0/720.0 * k * F(U.row(i+3)) 
			- 264.0/720.0 * k * F(U.row(i+2)) 
			+ 106.0/720.0 * k * F(U.row(i+1)) 
			- 19.0/720.0 * k * F(U.row(i)) 
			,251.0/720.0 * k);
	}
	
	return U;
}

MatrixXf TRBDF2Interation(MatrixXf UN, float k){
 	MatrixXf result;
	int n = UN.cols();
	//the function can be defined here *
	//float mu = 1/81.45;
	result = Eigen::Matrix<float,Dynamic,Dynamic>();
	result.resize(1,n);

	MatrixXf y1 = F(UN);

	// solve y2 = F(UN + k / 4.0 * (y1 + y2) )
	// X2 = UN + k / 4.0 * (y1 + y2)  
	// (X2 - UN - k / 4.0 * y1) / (4.0 / k) = y2
	// (X2 - UN - k / 4.0 * y1) / (4.0 / k)  =  F(X2)
	// X2  =  (4.0 / k) * F(X2) + UN + k / 4.0 * y1
	MatrixXf X2 = BDF(UN + (k / 4.0) * y1, k / 4.0);
	MatrixXf y2 = (X2 - UN - (k / 4.0) * y1) / (k / 4.0);
	MatrixXf X3 = BDF(UN + (k / 3.0) * (y1 + y2), k / 3.0);
	MatrixXf y3 = (X3 - UN) / (k / 3.0) - y1 - y2;
	result = UN + 1.0/3.0 * k * (y1 + y2 + y3);
	return result;
}

MatrixXf TRBDF2Method(MatrixXf U, float k, int N){
	for (int i = 0; i < N; i++) U.row(i+1) = TRBDF2Interation(U.row(i),k);
	return U;
}

MatrixXf ESDIRKInteration(MatrixXf UN, float k){
 	MatrixXf result;
	int n = UN.cols();
	//the function can be defined here *
	//float mu = 1/81.45;
	result = Eigen::Matrix<float,Dynamic,Dynamic>();
	result.resize(1,n);

	MatrixXf y1 = F(UN);

	// solve y2 = F(UN + k / 4.0 * (y1 + y2) )
	// X2 = UN + k / 4.0 * (y1 + y2)  
	// (X2 - UN - k / 4.0 * y1) / (4.0 / k) = y2
	// (X2 - UN - k / 4.0 * y1) / (4.0 / k)  =  F(X2)
	// X2  =  (4.0 / k) * F(X2) + UN + k / 4.0 * y1
	MatrixXf X2 = BDF(UN + (k / 4.0) * y1, k / 4.0);
	MatrixXf y2 = (X2 - UN - (k / 4.0) * y1) / (k / 4.0);

	float a31 = 8611.0/62500.0;
	float a32 = - 1743.0/31250.0;
	float a33 = 1.0/4.0;

	MatrixXf X3 = BDF(UN + k * (a31 * y1 + a32 * y2), k * a33);
	MatrixXf y3 = (X3 - UN - k * (a31 * y1 + a32 * y2) )/ (k * a33);

	float a41 = 5012029.0/34652500.0;
	float a42 = - 654441.0/2922500.0;
	float a43 = 174375.0/388108.0;
	float a44 = 1.0/4.0;

	MatrixXf X4 = BDF(UN + k * (a41 * y1 + a42 * y2 + a43 * y3), k * a44);
	MatrixXf y4 = (X4 - UN - k * (a41 * y1 + a42 * y2 + a43 * y3) )/ (k * a44);

	float a51 = 15267082809.0/155376265600.0;
	float a52 = - 71443401.0/120774400.0;
	float a53 = 730878875.0/902184768.0;
	float a54 = 2285395.0/8070912.0;
	float a55 = 1.0/4.0;


	MatrixXf X5 = BDF(UN + k * (a51 * y1 + a52 * y2 + a53 * y3 + a54 * y4), k * a55);
	MatrixXf y5 = (X5 - UN - k * (a51 * y1 + a52 * y2 + a53 * y3 + a54 * y4) )/ (k * a55);

	float a61 = 82889.0/524892.0;
	float a62 = 0.0;
	float a63 = 15625.0/83664.0;
	float a64 = 69875/102672;
	float a65 = - 2260.0/8211.0;
	float a66 = 1.0/4.0;

	MatrixXf X6 = BDF(UN + k * (a61 * y1 + a62 * y2 + a63 * y3 + a64 * y4 + a65 * y5), k * a66);
	MatrixXf y6 = (X6 - UN - k * (a61 * y1 + a62 * y2 + a63 * y3 + a64 * y4 + a65 * y5) )/ (k * a66);

	result = UN + k * 
	(82889.0/524892.0 *  y1 
		+ 15625.0/83664.0 * y3 
		+ 69875.0/102672.0 * y4 
		- 2260.0/8211.0 * y5 
		+ 1.0/4.0 * y6);

	return result;
}

MatrixXf ESDIRKMethod(MatrixXf U, float k, int N){
	for (int i = 0; i < N; i++) U.row(i+1) = ESDIRKInteration(U.row(i),k);
	return U;
}

MatrixXf GLRK1Method(MatrixXf U, float k, int N){
	for(int i = 0; i < N; i++){
		MatrixXf mid = BDF(U.row(i),0.5 * k);
		U.row(i+1) = 2.0 * mid - U.row(i);
	}
}

// RowVectorXd  GLRK2_solve(RowVectorXd joined, float k, int n){
// 	// 
// 	RowVectorXd next(2 * n);
// 	next = RowVectorXd::Zero();
// 	while(1){
// 		next.head(n) = k * F(next.head(n) ) + joined.head(n);
// 		next.tail(n) = k * F(next.tail(n) ) + joined.tail(n);	

// 		if(	( (next.head(n) - k * F(next.head(n) ) - joined.head(n)).norm() < eps ) 
// 			&& 
// 			( (next.tail(n) - k * F(next.tail(n) ) - joined.tail(n)).norm() < eps )
// 			)
// 			break;
// 	}
// 	return next;


// }


// MatrixXf GLRK2Method(MatrixXf U, float k, int N){
// 	Matrix2f a;
// 	a << 1.0/4.0, (1.0/12.0) * (3.0 - 2.0 * pow(3.0,0.5)),
// 		(1.0/12.0) * (3.0 + 2.0 * pow(3.0,0.5)), 1.0/4.0;
// 	int n = UN.cols();

	

// 	for(int i = 0; i < 1; i++){
// 		U.row(i+1) = ClassicRKInteration(U.row(i),k);
// 	}
// 	for (int i = 0; i < N; i++){
// 		RowVectorXd joined(2 * n);
// 		RowVectorXd y1(n) = U.row(i)
// 		RowVectorXd y2(n) = U.row(i+1);
// 		joined << y1, y2;
// 		joined  = GLRK2_solve(joined, k, n);
// 		U.row(i) = joined.head(n);
// 		U.row(i+1) = joined.tail(n);
// 	}
// 	return U;
// }





int main(){	

	int  n; // n 是微分方程的维度
	float T; // T is the period
	float k; // 步长
	int N; // 总迭代次数
	float mu;
	clock_t start,end;

	// ------ test 1 -------  ///
	n = 6;
	// T = 17.06521656015796;
	T = 19.14045706162071;
	N = 24000;
	k = T/N;
	mu = 1/81.45;


	// n = 1;
	// T = 2;
	// N = 4;
	// k = T/N;



//---- Initiate matrices ----------//
	
	MatrixXf U; 
	U = Eigen::Matrix<float,Dynamic,Dynamic>();
	U.resize(N+1,n);


//----- Initiate u ----------------------//
	// U(0,0) = 0.994; //例子：初始化
	// U(0,1) = 0; //
	// U(0,2) = 0; //
	// U(0,3) = 0; //
	// U(0,4) = -2.0015851063790825224; //
	// U(0,5) = 0; //

	U(0,0) = 0.87978; //例子：初始化
	U(0,1) = 0; //
	U(0,2) = 0; //
	U(0,3) = 0; //
	U(0,4) = -0.3797; //
	U(0,5) = 0; //


	start = clock();
	U = EulerMethod(U,k,N);
 	// U = ClassicRKMethod(U,k,N);
	// U = AB1Method(U,k,N);
	// U = AB2Method(U,k,N);
	// U = AB3Method(U,k,N);
	// U = AB4Method(U,k,N);
	// U = AM2Method(U,k,N);
	// U = AM3Method(U,k,N);
	// U = AM4Method(U,k,N);
	// U = AM5Method(U,k,N);
	// U = BDF1Method(U,k,N);
	// U = BDF2Method(U,k,N);
	// U = BDF3Method(U,k,N);
	// U = BDF4Method(U,k,N);
	// U = TRBDF2Method(U,k,N);
	// U = ESDIRKMethod(U,k,N);	
	// U = GLRK1Method(U,k,N);
	end = clock();   

	std::cout << "Error = " << (U.row(N) - U.row(0)).norm()<<endl;

	std::cout << "Accuracy Order = " << - log(k)/log((U.row(N) - U.row(0)).norm()) <<endl;

	cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl;  //输出时间（单位：ｓ）


	ofstream outfile;
	outfile.open("example.txt");
	outfile << U.leftCols(2)<< endl;
	outfile.close();


}