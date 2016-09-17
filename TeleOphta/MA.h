#ifndef __MA_h
#define __MA_h

#include <stdio.h>
#include <iostream>

#include "TOMorph.h"
#include "basicOperations.h"
#include "maxTree.h"
#include "fastMorph.h"
#include "detectROI.h"
#include "TOfft.h"
#include "Gabor.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;



class HMCand{
public:
	int posit[2];
	list<int> p[2];

	HMCand();
};

HMCand::HMCand(){
	posit[0] = 0;
	posit[1] = 0;

}




class MACand{
public:
	int labelref;
	int center[2];
	list<int> p[2];
	int minX,maxX,minY,maxY;
	int W,H;
	float WH;
	int length, length2;
	float circ;
	int isGT;

	float maxRes;
	float meanRes;
	float meanVessel;
	float inVessel;
	float ratioMeanVessel;
	float ratioInVessel;
	int area;

	int n_winS_thL_ccS,n_winS_thL_ccL,n_winL_thL_ccS,n_winL_thL_ccL,
		n_winS_thH_ccS,n_winS_thH_ccL,n_winL_thH_ccS,n_winL_thH_ccL;

	int p1[2];
	int p2[2];

	MACand();
	void vesselAnalyse(Mat imASF, Mat maCandiRaw, Mat imMainVesselASF, Mat imMainVesselOrient, Mat imlabel, int *imInfo );
	void geoLength(Mat imin, Mat imPerimeter, Mat imstate);
	void envAnalyse(Mat imASF, Mat maCandiRaw, int* imInfo);
};

MACand::MACand(){
	center[0] = 0; center[1] = 0;
	meanRes = 0; meanVessel = 0; maxRes = 0;
	inVessel = 0;
	area = 0;
	W=0; H=0;
	minX=99999; minY=99999; maxX=-1; maxY=-1;
	length=0; length2=0;
	isGT=0;
	n_winS_thL_ccS=0; n_winS_thL_ccL=0; n_winL_thL_ccS=0; n_winL_thL_ccL=0;
	n_winS_thH_ccS=0; n_winS_thH_ccL=0; n_winL_thH_ccS=0; n_winL_thH_ccL=0;;
}

void MACand::vesselAnalyse(Mat imASF, Mat maCandiRaw, Mat imMainVesselASF, Mat imMainVesselOrient, Mat imlabel, int *imInfo){
	// parameters:
	int windowSize = imInfo[2]*3; // This is half size
	//=================
	// 1. nearest vessel mean value 
	int startX = center[0]-windowSize;
	int startY = center[1]-windowSize;
	int endX = center[0]+windowSize;
	int endY = center[1]+windowSize;
	if (startX<0) startX=0;
	if (startY<0) startY=0;
	if (endX>=imASF.cols) endX=imASF.cols-1;
	if (endY>=imASF.rows) endY=imASF.rows-1;

	/*cout<<startX<<" "<<endX<<" "<<startY<<" "<<endY<<" ";*/
	int np(0);
	for (int j=startY; j<=endY; j++){
		for (int i=startX; i<=endX; i++){
			if (imlabel.at<int>(j,i) == (labelref+1)) continue;
			if (imMainVesselASF.at<uchar>(j,i)>0){
				meanVessel += imASF.at<uchar>(j,i);
				np++;
			}
		}
	}
	if (np==0) meanVessel = 0;
	else meanVessel /= np;
	
	// 2. is in the vessel
	windowSize = imInfo[2];
	startX = center[0]-windowSize;
	startY = center[1]-windowSize;
	endX = center[0]+windowSize;
	endY = center[1]+windowSize;
	if (startX<0) startX=0;
	if (startY<0) startY=0;
	if (endX>=imASF.cols) endX=imASF.cols-1;
	if (endY>=imASF.rows) endY=imASF.rows-1;

	list<int>::iterator it1;
	list<int>::iterator it2;
	it1 = p[0].begin();
	it2 = p[1].begin();
	int orient[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
	while (it1 != p[0].end()){
		orient[imMainVesselOrient.at<uchar>(*it2, *it1)-1]++;
		it1++;
		it2++;
	}
	int maxOriV(0), maxOriP(0);
	for (int i=0; i<12; i++){
		if (orient[i] > maxOriV){
			maxOriV = orient[i];
			maxOriP = i;
		}
	}
	
	np =0;
	double intv = PI/12;
	double mapAng[12] = {-intv*6,-intv*5,-intv*4,-intv*3,-intv*2,-intv,0,intv,intv*2,intv*3,intv*4,intv*5};
	if (maxOriV!=0){
		int nb[4];
		nb[0] = (maxOriP+1+12)%12+1;
		nb[1] = (maxOriP+2+12)%12+1;
		nb[2] = (maxOriP-1+12)%12+1;
		nb[3] = (maxOriP-2+12)%12+1;
		double angCenter = mapAng[maxOriP];

		for (int j=startY; j<=endY; j++){
			for (int i=startX; i<=endX; i++){
				if (imlabel.at<int>(j,i) == (labelref+1)) continue;
				if (imMainVesselOrient.at<uchar>(j,i)==(maxOriP+1) || imMainVesselOrient.at<uchar>(j,i)==nb[0]
				|| imMainVesselOrient.at<uchar>(j,i)==nb[1] || imMainVesselOrient.at<uchar>(j,i)==nb[2]
				|| imMainVesselOrient.at<uchar>(j,i)==nb[3] ){
					// if the point is in the same direction
					double angNow = getAngle(center[0],center[1],i,j);
					double angDiff = abs(angNow-angCenter);
					if (angDiff>PI/2) angDiff = PI - angDiff;
					if (angDiff>PI/6) continue;

					if (imASF.at<uchar>(j,i)>0){
						inVessel += imASF.at<uchar>(j,i);
						np++;
					}
				}
			}
		}

		if (np==0) inVessel = 0;
		else inVessel /= np;
	}
	
	else inVessel = 0;

	if (meanVessel==0) ratioMeanVessel = 255;
	else ratioMeanVessel = meanRes/meanVessel;
	if (inVessel==0) ratioInVessel = 255;
	else ratioInVessel = meanRes/inVessel;
}

void MACand::geoLength(Mat imin, Mat imPerimeter, Mat imstate){
	int se(6);
	int **se_even = nl(se,1);
	int **se_odd = nl(se,0);
	int size[2] = {imPerimeter.cols, imPerimeter.rows};

	// 1. Get border pixels
	list<int>::iterator it1; 
	list<int>::iterator it2;
	it1 = p[0].begin();
	it2 = p[1].begin();
	queue<int> Q[2];
	while (it1!=p[0].end()){		
		if (imPerimeter.at<uchar>(*it2,*it1)>0){
			Q[0].push(*it1);
			Q[1].push(*it2);
		}
		imstate.at<uchar>(*it2,*it1) = 0;
		it1++;
		it2++;
	}
	// 2. Get the pixel most far 
	int px,py,mx,my,len(0);
	float dist, maxDist(0);
	while(!Q[0].empty()){
		px = Q[0].front();
		py = Q[1].front();
		dist = sqrt((float)(center[0]-px)*(center[0]-px) + (center[1]-py)*(center[1]-py));
		if (dist>=maxDist){
			maxDist = dist;
			mx = px;
			my = py;
		}
		Q[0].pop();
		Q[1].pop();
	}
	Q[0].push(mx);
	Q[1].push(my);

	while(!Q[0].empty()){
		mx = Q[0].front();
		my = Q[1].front();
		if (imstate.at<uchar>(my,mx)!=0){
			Q[0].pop();
			Q[1].pop();
			continue;
		}

		for (int k=0; k<se; ++k){
			if (my%2==0){
				px = mx + se_even[k][0];
				py = my + se_even[k][1];
			}
			else{
				px = mx+ se_odd[k][0];
				py = my + se_odd[k][1];
			}
			if (px<0 || px>=size[0] || py<0 || py>=size[1]) continue;
			if (imin.at<uchar>(py,px)>0 && imstate.at<uchar>(py,px)==0){  // see if it's on the edge;
				Q[0].push(px);
				Q[1].push(py);
			}
		}
		imstate.at<uchar>(my,mx) = 1;
		Q[0].pop();
		Q[1].pop();
	}
	Q[0].push(mx);
	Q[1].push(my);
	Q[0].push(-1); // -1 is a mark point
	Q[1].push(-1);
	imstate.at<uchar>(my,mx) = 2;
	p1[0] = mx;
	p1[1] = my;

	// 4. Second propagation
	while(!Q[0].empty()){
		mx = Q[0].front();
		my = Q[1].front();

		if (mx == -1) {  // if the mark point pop out, one iteration is done, len ++
			++len;
			Q[0].pop();
			Q[1].pop();
			if (Q[0].empty()) break;
			Q[0].push(-1);
			Q[1].push(-1);
			mx = Q[0].front();
			my = Q[1].front();
		}
		p2[0] = mx;
		p2[1] = my;

		for (int k=0; k<se; ++k){
			if (my%2==0){
				px = mx + se_even[k][0];
				py = my + se_even[k][1];
			}
			else{
				px = mx + se_odd[k][0];
				py = my + se_odd[k][1];
			}
			if (px<0 || px>=size[0] || py<0 || py>=size[1]) continue;
			if (imin.at<uchar>(py,px)>0 && imstate.at<uchar>(py,px)==1){
				Q[0].push(px);	
				Q[1].push(py);
				imstate.at<uchar>(py,px) = 2;
			}
		}

		//	imstate[my][mx] = 2;
		Q[0].pop();
		Q[1].pop();
	}

	length = len;
	length2 = sqrt(pow(float(p1[0]-p2[0]),2) + pow(float(p1[1]-p2[1]),2));

	circ = (float) (4*area)/(3.1415*length*length);

	for (int i=0; i<se; i++){
		delete[] se_even[i];
		delete[] se_odd[i];
	}
	delete[] se_even;
	delete[] se_odd;
}
	

void MACand::envAnalyse(Mat imASF, Mat maCandiRaw, int* imInfo){
	/************************************************************************/
	/* Analyse the number of the connected components around the candidate
	1. In a small window, number of CC whos area between imInfo[3]^2/3~2*imInfo[3]^2
	2. In a large window, number of CC whos area between ns3 2*imInfo[3]^2 ~ inf
	3. Thresholds are maxRes and meanRes. (H and L) */
	/************************************************************************/
	int size[2] = {imASF.cols, imASF.rows};
	int W1 = imInfo[3];
	int W2 = imInfo[3]*2;
	int W3 = imInfo[3]*4;
	int C_area1 = imInfo[3]*imInfo[3]/3;
	int C_area2 = imInfo[3]*imInfo[3]*2;

	Mat imASFCut = Mat::zeros(W3*2+1,W3*2+1,CV_8U);
	Mat imCandiCut = Mat::zeros(W3*2+1,W3*2+1,CV_8U);
	Mat imtempCut1 = imASFCut.clone();
	Mat imtempCut2 = imASFCut.clone();
	Mat imtemp32 = Mat::zeros(W3*2+1,W3*2+1,CV_32S);

	int s,t;
	for (int n=-W3; n<=W3; ++n){
		for (int m=-W3; m<=W3; ++m){
			s = center[0]+m;
			t = center[1]+n;
			if (s<0 || s>=size[0] || t<0 || t>=size[1]) continue;
			imASFCut.at<uchar>(n+W3,m+W3) = imASF.at<uchar>(t,s);
			imCandiCut.at<uchar>(n+W3,m+W3) = maCandiRaw.at<uchar>(t,s);
		}
	}

	// Higher threshold 
	threshold(imASFCut,imtempCut1,maxRes-1,255,0);
	subtract(imtempCut1,imCandiCut,imtempCut2);
	Label(imtempCut2,imtemp32,6);
	int N = labelCount(imtemp32);
	int *ccAreaCount = new int[N];
	int *ccVisited = new int[N];
	memset(ccAreaCount,0,sizeof(int)*N);
	memset(ccVisited,0,sizeof(int)*N);
	
	for(int j=0; j<W3*2+1; j++){
		for (int i=0; i<W3*2+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			ccAreaCount[imtemp32.at<int>(j,i)-1]++;
		}
	}

	for(int j=W1*3; j<W1*5+1; j++){
		for (int i=W1*3; i<W1*5+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			if (ccVisited[imtemp32.at<int>(j,i)-1]==1) continue;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>=C_area1 && ccAreaCount[imtemp32.at<int>(j,i)-1]<=C_area2) n_winS_thH_ccS++;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>C_area2) n_winS_thH_ccL++;
			ccVisited[imtemp32.at<int>(j,i)-1]=1;
		}
	}

	memset(ccVisited,0,sizeof(int)*N);
	for(int j=W1*2; j<W1*6+1; j++){
		for (int i=W1*2; i<W1*6+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			if (ccVisited[imtemp32.at<int>(j,i)-1]==1) continue;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>=C_area1 && ccAreaCount[imtemp32.at<int>(j,i)-1]<=C_area2) n_winL_thH_ccS++;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>C_area2) n_winL_thH_ccL++;
			ccVisited[imtemp32.at<int>(j,i)-1]=1;
		}
	}
	delete[] ccAreaCount;
	delete[] ccVisited;
	
	// lower threshold
	threshold(imASFCut,imtempCut1,meanRes-1,255,0);
	subtract(imtempCut1,imCandiCut,imtempCut2);
	Label(imtempCut2,imtemp32,6);
	N = labelCount(imtemp32);
	ccAreaCount = new int[N];
	ccVisited = new int[N];
	memset(ccAreaCount,0,sizeof(int)*N);
	memset(ccVisited,0,sizeof(int)*N);

	for(int j=0; j<W3*2+1; j++){
		for (int i=0; i<W3*2+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			ccAreaCount[imtemp32.at<int>(j,i)-1]++;
		}
	}

	for(int j=W1*3; j<W1*5+1; j++){
		for (int i=W1*3; i<W1*5+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			if (ccVisited[imtemp32.at<int>(j,i)-1]==1) continue;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>=C_area1 && ccAreaCount[imtemp32.at<int>(j,i)-1]<=C_area2) n_winS_thL_ccS++;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>C_area2) n_winS_thL_ccL++;
			ccVisited[imtemp32.at<int>(j,i)-1]=1;
		}
	}

	memset(ccVisited,0,sizeof(int)*N);
	for(int j=W1*2; j<W1*6+1; j++){
		for (int i=W1*2; i<W1*6+1; i++){
			if (imtemp32.at<int>(j,i)==0) continue;
			if (ccVisited[imtemp32.at<int>(j,i)-1]==1) continue;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>=C_area1 && ccAreaCount[imtemp32.at<int>(j,i)-1]<=C_area2) n_winL_thL_ccS++;
			if (ccAreaCount[imtemp32.at<int>(j,i)-1]>C_area2) n_winL_thL_ccL++;
			ccVisited[imtemp32.at<int>(j,i)-1]=1;
		}
	}

	delete[] ccAreaCount;
	delete[] ccVisited;

}



void writeFile(MACand *maCandList, int N, int* imInfo){
	/************************************************************************/
	/*	1. maxRes
		2. meanRes
		3. meanRes/meanVessel
		4. meanRes/inVessel
		5. min(W,H)/max(W,H)
		6. area
		7. length
		8 length2/length
		9. circularity
		10~17. n_winS_thL_ccS,n_winS_thL_ccL,n_winL_thL_ccS,n_winL_thL_ccL,
		n_winS_thH_ccS,n_winS_thH_ccL,n_winL_thH_ccS,n_winL_thH_ccL
		18. isGT
		*/
	/************************************************************************/
	ofstream myfile;
	myfile.open("maFeatures.txt");

	float sizeNorm1=imInfo[3];
	float sizeNorm2=imInfo[3]*imInfo[3];

	for (int i=0; i<N; i++){
		myfile << maCandList[i].center[0]<<" "<< maCandList[i].center[1]<<" "<<maCandList[i].maxRes<<" "<<maCandList[i].meanRes<<" "
			<< maCandList[i].ratioMeanVessel<<" "<<maCandList[i].ratioInVessel<<" "<<maCandList[i].WH<<" "<<(float(maCandList[i].area)/sizeNorm2)<< " "
			<<(float(maCandList[i].length)/sizeNorm1)<<" "<<float(maCandList[i].length2)/maCandList[i].length<<" "<<maCandList[i].circ<<" "<<
			maCandList[i].n_winS_thL_ccS<<" "<<maCandList[i].n_winS_thL_ccL<<" "<<maCandList[i].n_winL_thL_ccS<<" "<<
			maCandList[i].n_winL_thL_ccL<<" "<<maCandList[i].n_winS_thH_ccS<<" "<<maCandList[i].n_winS_thH_ccL<<" "<<
			maCandList[i].n_winL_thH_ccS<<" "<<maCandList[i].n_winL_thH_ccL<<" "<<maCandList[i].isGT<<"\n";
	}
	myfile.close();
}



MACand* ccAnalyseMA(Mat maCandiRaw, Mat imgreen, Mat imASF, Mat imROI, Mat imMainVesselGB, 
	Mat imMainVesselASF, Mat imMainVesselOrient, Mat imMainVesselSup, Mat imGT, int* imInfo){
		
                cout<<"(DBG) Pass here 1"<<endl;
		int imSize[2] = {imgreen.rows,imgreen.cols};

		Mat imMaxiG = imgreen.clone();
		Mat imtemp1 = imgreen.clone();
		Mat imtemp2 = imgreen.clone();
		Mat imMaASF = imgreen.clone();
		Mat imPerimeter = imgreen.clone();
		Mat imtemp32 = Mat::zeros(imSize[0], imSize[1], CV_32S);

		imCompare(maCandiRaw,0,1,imASF,0,imMaASF);

		Erode(maCandiRaw,imtemp1,6,1);
		subtract(maCandiRaw,imtemp1,imPerimeter);

		// 1. maximas value in residu : maxRes
		Maxima(imASF,imtemp1,6);
		imCompare(imtemp1,0,1,imASF,0,imMaxiG);
	
		fastDilate(imGT,imtemp2,6,imInfo[3]/3);
		
		Label(maCandiRaw,imtemp32,6);
		int N = labelCount(imtemp32);
		cout<<N<<endl;
                cout<<"(DBG) Pass here 2"<<endl;

		MACand *maCandList = new MACand[N];

		for (int j=0; j<imgreen.rows; j++){
			for (int i=0; i<imgreen.cols; i++){
				if (imtemp32.at<int>(j,i) == 0) continue;
				maCandList[imtemp32.at<int>(j,i)-1].p[0].push_back(i);
				maCandList[imtemp32.at<int>(j,i)-1].p[1].push_back(j);
				maCandList[imtemp32.at<int>(j,i)-1].center[0] += i;
				maCandList[imtemp32.at<int>(j,i)-1].center[1] += j;
				maCandList[imtemp32.at<int>(j,i)-1].meanRes += imASF.at<uchar>(j,i);
				maCandList[imtemp32.at<int>(j,i)-1].area ++;
				if (maCandList[imtemp32.at<int>(j,i)-1].minX>=i) maCandList[imtemp32.at<int>(j,i)-1].minX=i;
				if (maCandList[imtemp32.at<int>(j,i)-1].minY>=j) maCandList[imtemp32.at<int>(j,i)-1].minY=j;
				if (maCandList[imtemp32.at<int>(j,i)-1].maxX<=i) maCandList[imtemp32.at<int>(j,i)-1].maxX=i;
				if (maCandList[imtemp32.at<int>(j,i)-1].maxY<=j) maCandList[imtemp32.at<int>(j,i)-1].maxY=j;

				if (maCandList[imtemp32.at<int>(j,i)-1].maxRes < imMaxiG.at<uchar>(j,i)) 
					maCandList[imtemp32.at<int>(j,i)-1].maxRes = imMaxiG.at<uchar>(j,i);
				
				if (imtemp2.at<uchar>(j,i)>0) maCandList[imtemp32.at<int>(j,i)-1].isGT=1;
			}
		}
                cout<<"(DBG) Pass here 3"<<endl;

		
		for (int i=0; i<N; i++){
			maCandList[i].labelref = i;
			maCandList[i].meanRes /= maCandList[i].area;
			maCandList[i].center[0] /= maCandList[i].area;
			maCandList[i].center[1] /= maCandList[i].area;
			maCandList[i].H = maCandList[i].maxY - maCandList[i].minY + 1;
			maCandList[i].W = maCandList[i].maxX - maCandList[i].minX + 1;
			
			if (maCandList[i].H > maCandList[i].W) maCandList[i].WH = float(maCandList[i].W)/maCandList[i].H;
			else maCandList[i].WH = float(maCandList[i].H)/maCandList[i].W;

                        // maCandList[i].vesselAnalyse(imASF,maCandiRaw,imMainVesselASF,imMainVesselOrient,imtemp32,imInfo);
                        // maCandList[i].geoLength(imMaASF,imPerimeter,imtemp1);

                        // maCandList[i].envAnalyse(imASF,maCandiRaw,imInfo);
		}

	

                cout<<"(DBG) Pass here 4"<<endl;


		writeFile(maCandList,N,imInfo);
		delete[] maCandList;
		/*	int nn = 603;
		cout<<maCandList[nn].p[0].front()<<" "<<maCandList[nn].p[1].front()<<" "<<maCandList[nn].center[0]<<" "
		<<maCandList[nn].center[1]<<" "<<maCandList[nn].H<<" "<<maCandList[nn].W<<" "<<maCandList[nn].maxRes<<" "
		<<maCandList[nn].meanRes<<" "<<maCandList[nn].area<<" "<<maCandList[nn].meanVessel<<" "
		<<maCandList[nn].inVessel<<endl;

		float r1,r2;
		imtemp1.setTo(0);
		imtemp2.setTo(0);
		for (int j=0; j<imgreen.rows; j++){
		for (int i=0; i<imgreen.cols; i++){
		if (imtemp32.at<int>(j,i) == 0) continue;
		if (maCandList[imtemp32.at<int>(j,i)-1].meanVessel ==0 ) r1 = 999;
		else r1 = maCandList[imtemp32.at<int>(j,i)-1].meanRes / maCandList[imtemp32.at<int>(j,i)-1].meanVessel;
		if (r1>=3) imtemp1.at<uchar>(j,i) = 255;
		else imtemp1.at<uchar>(j,i) = r1/3*255;

		if (maCandList[imtemp32.at<int>(j,i)-1].inVessel ==0 ) r2 = 999;
		else r2 = maCandList[imtemp32.at<int>(j,i)-1].meanRes / maCandList[imtemp32.at<int>(j,i)-1].inVessel;
		if (r2>=3) imtemp2.at<uchar>(j,i) = 255;
		else imtemp2.at<uchar>(j,i) = r2/3*255;
		}
		}

		imwrite("z3.png",imtemp1);
		imwrite("z4.png",imtemp2);
		threshold(imtemp1,imtemp3,69,255,0);
		threshold(imtemp2,imtemp4,254,255,0);
		imwrite("z5.png",imtemp3);
		imwrite("z6.png",imtemp4);
		imInf(imtemp4,imtemp3,imtemp4);
		imwrite("z7.png",imtemp4);*/
		
	
		//Mat imtemp1 = imgreenR.clone();
		//Mat imtemp2 = imgreenR.clone();
		//Mat imtemp3 = imgreenR.clone();
		//
		//threshold(imMainVesselGB,imtempR1,0,255,0);
		//RecUnderBuild(imtempR3,imtempR1,imtempR2,6);
		//imwrite("z3.png",imtempR2);
		//subtract(imtempR3,imtempR2,imtempR1);
		//imwrite("z4.png",imtempR1);
}



void detectMA(Mat imgreenR, Mat imROIR, Mat imASFR, Mat imBorderReflectionR, Mat imBrightPart,
	vector<Mat> vessels, vector<Mat> vesselProperty, int* imInfoR, int* ODCenter, Mat imGT){

	int imSizeR[2] = {imgreenR.rows,imgreenR.cols};
	Mat imtempR1 = imgreenR.clone();
	Mat imtempR2 = imgreenR.clone();
	Mat imtempR3 = imgreenR.clone();
	Mat imMainVesselGB = imgreenR.clone();
	Mat imMainVesselOrient = imgreenR.clone();
	Mat imMainVesselSup = imgreenR.clone();
	Mat imMainVesselASF = imgreenR.clone();
	Mat maCandiRaw = imgreenR.clone();
	Mat imtemp32int = Mat::zeros(imSizeR[0], imSizeR[1], CV_32S);
	Mat imMACandi = imgreenR.clone();
	Mat imBN = imgreenR.clone();
	Mat imDarkPart = imgreenR.clone();
	Mat imHMCandi = Mat::zeros(imSizeR[0], imSizeR[1], CV_8U);
	Mat imROIRs = imgreenR.clone();
	Erode(imROIR,imROIRs,6,2);

	//========================================================
	// Reconstruction of OD
	Mat imOD=Mat::zeros(imgreenR.rows,imgreenR.cols,CV_8U);
	Mat imMC=Mat::zeros(imgreenR.rows,imgreenR.cols,CV_8U);
	int ODCenterL[2]={-1,-1}, MCCenterL[2]={-1,-1};
	if(ODCenter[0]!=-1){
		float f = imgreenR.cols/512.0f;
		ODCenterL[0] = ODCenter[0]*f; ODCenterL[1] = ODCenter[1]*f;
		MCCenterL[0] = ODCenter[2]*f; MCCenterL[1] = ODCenter[3]*f;
		Point center(ODCenter[0]*f,ODCenter[1]*f);
		circle(imOD,center,imInfoR[1]/3*2,255,-1);
		center.x = MCCenterL[0];
		center.y = MCCenterL[1]; 
		circle(imMC,center,imInfoR[1]/4,255,-1);
	}
	cout<<"Large ODCenter: "<<ODCenterL[0]<<" "<<ODCenterL[1]<<endl;
	cout<<"Large MCCenter: "<<MCCenterL[0]<<" "<<MCCenterL[1]<<endl;
	//========================================================

	
	//========================================================f
	// Bright noise
	int meanV = meanValue(imgreenR,1);// fill background
	imCompare(imROIR,0,0,meanV,imgreenR,imtempR1);
	fastMeanFilter(imtempR1,imInfoR[1]/3*2,imtempR2);
	subtract(imgreenR,imtempR2,imtempR1);
	fastMeanFilter(imtempR1,imInfoR[3]*3,imtempR2);
	imCompare(imROIR,0,0,0,imtempR2,imBN);

	imtempR3= Mat::zeros(imtempR1.rows,imtempR1.cols,CV_8U);
	Point center(imgreenR.cols/2,imgreenR.rows/2);
	circle(imtempR3,center,imInfoR[1],255,-1);
	imSup(imtempR3,imOD,imtempR1);
	RecUnderBuild(imgreenR,imtempR1,imtempR2,6);
	subtract(imgreenR,imtempR2,imtempR3);
	RecUnderBuild(imtempR3,imBorderReflectionR, imtempR1,6);
	threshold(imtempR1,imtempR2,4,255,0);
	imSup(imtempR2,imBorderReflectionR,imBorderReflectionR);
	//========================================================

	


	//========================================================
	// Gabor vessel analysis
        if (0){
            cout<<"(DBG) PASS HERE"<<endl;
            vesselAnalyseGabor(imASFR,imMainVesselGB,imMainVesselOrient, imMainVesselSup, imMainVesselASF, imInfoR);
            float meanVessel(0), np(0);
            for (int j=0; j<imASFR.rows; j++){
                for (int i=0; i<imASFR.cols; i++){
                    if (imMainVesselASF.at<uchar>(j,i) == 0) continue;
                    meanVessel += imMainVesselASF.at<uchar>(j,i);
                    np++;
                }
            }
            if (np==0) meanVessel=0;
            else meanVessel /= np;
            cout<<"mean vessel: "<<meanVessel<<endl;
        }
	//========================================================



	//========================================================
	// Get candidates
	/************************************************************************/
	/*	1. divided by 2, underbuild, get maximas
		2. remove large elements
		3. remove by mean value in each CC
		4. remove too long too small things.	*/
	/************************************************************************/
        int C_area_min = imInfoR[3]; // imInfoR[3]/2;
	if (C_area_min<4) C_area_min=4;

	divide(imASFR,2,imtempR1);
	RecUnderBuild(imASFR,imtempR1,imtempR2,6);
	Maxima(imtempR2,imtempR3,6);
	binAreaSelection(imtempR3,imtempR1,6,imInfoR[3]*imInfoR[3]);
	LabelByMean(imtempR1,imASFR,imtempR2,6);
	threshold(imtempR2,imtempR3,4,255,0);
	lengthOpening(imtempR3,imtempR1,imInfoR[3]*1.5,imInfoR[3]*imInfoR[3],0,2);
	subtract(imtempR3,imtempR1,imtempR2);
	binAreaSelection(imtempR2,imtempR1,6,C_area_min);
	subtract(imtempR2,imtempR1,imtempR3);

	RecUnderBuild(imtempR3,imOD,imtempR1,6);
	subtract(imtempR3,imtempR1,maCandiRaw);
	imwrite("imMaCandi.png",maCandiRaw);

        RecUnderBuild(imASFR,maCandiRaw,imtempR3,6);
        imwrite("imASFRec.png",imtempR3);


	ccAnalyseMA(maCandiRaw, imgreenR, imASFR, imROIR, imMainVesselGB, imMainVesselASF, imMainVesselOrient, imMainVesselSup, imGT, imInfoR);

	

	
	//subtract(vessels[2],vessels[0],imtempR1);
	//imCompare(imROIRs,0,0,0,imtempR1,imtempR1);
	//// remove od region
	//RecUnderBuild(imtempR1,imOD,imtempR3,6);
	//subtract(imtempR1,imtempR3,imMACandi);
	//imwrite("imMACandi.png",imMACandi);
	//========================================================


	cout<<"Hello world"<<endl;
}



#endif
