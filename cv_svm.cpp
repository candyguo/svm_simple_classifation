// cv_scan_directory.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "fstream"
#include <vector>
#include <string>
#include "iostream"
#include "svm.h"
using namespace std;

double inputArr[10][13]={
	1,0.708333,1,1,-0.320755,-0.105023,-1,1,-0.419847,-1,-0.225806,0,1, 

	-1,0.583333,-1,0.333333,-0.603774,1,-1,1,0.358779,-1,-0.483871,0,-1,

	1,0.166667,1,-0.333333,-0.433962,-0.383562,-1,-1,0.0687023,-1,-0.903226,-1,-1,

	-1,0.458333,1,1,-0.358491,-0.374429,-1,-1,-0.480916,1,-0.935484,0,-0.333333,

	-1,0.875,-1,-0.333333,-0.509434,-0.347032,-1,1,-0.236641,1,-0.935484,-1,-0.333333,

	-1,0.5,1,1,-0.509434,-0.767123,-1,-1,0.0534351,-1,-0.870968,-1,-1,

	1,0.125,1,0.333333,-0.320755,-0.406393,1,1,0.0839695,1,-0.806452,0,-0.333333,

	1,0.25,1,1,-0.698113,-0.484018,-1,1,0.0839695,1,-0.612903,0,-0.333333,

	1,0.291667,1,1,-0.132075,-0.237443,-1,1,0.51145,-1,-0.612903,0,0.333333,

	1,0.416667,-1,1,0.0566038,0.283105,-1,1,0.267176,-1,0.290323,0,1
};

double testArr[]={
	0.25,1,1,-0.226415,-0.506849,-1,-1,0.374046,-1,-0.83871,0,-1
};

void setSvmpara(struct svm_parameter *param)
{
	param->svm_type=C_SVC;
	param->kernel_type=RBF;
	param->degree=3;
	param->gamma=0;
	param->coef0=0;
	param->nu=0.5;
	param->cache_size=100;
	param->C=1;
	param->eps=1e-3;
	param->p=0.1;
	param->shrinking=1;
	param->probability=0;
	param->nr_weight=0;
	param->weight_label=NULL;
	param->weight=NULL;

}

void svmTraining()
{
	struct svm_parameter param;
	struct svm_problem prob;
	struct svm_model *model;
	struct svm_node *x_space;
	int cross_validation=0;
	int nr_fold=0;
	int sample_count=10;
	int feature_dim=12;
	setSvmpara(&param);
	prob.l=sample_count;
	prob.y=new double[sample_count];
	prob.x=new struct svm_node *[sample_count];
	for(int i=0;i<sample_count;i++)
	{
		prob.x[i]=new svm_node[feature_dim+1];
	}
	x_space=new struct svm_node[(feature_dim+1)*sample_count];
	for(int i=0;i<sample_count;i++)
	{
		prob.y[i]=inputArr[i][0];
	}
	/*int j = 0;

	for (int i=0; i<sample_count; i++)

	{
		prob.x[i] = &x_space[j];
		for (int k=0; k<feature_dim; k++)
		{
			x_space[i*feature_dim+k].index = k+1;
			x_space[i*feature_dim+k].value = inputArr[i][k+1];

		}
		x_space[(i+1)*feature_dim].index = -1;
		j = (i+1)*feature_dim + 1; 
	}*/
	for (int j=0; j<sample_count; j++)
	{
		for (int k=0; k<feature_dim+1; k++)
		{
			if(k==feature_dim)
			{
				prob.x[j][k].index=-1;
			}
			else{
			prob.x[j][k].index = k+1;
			prob.x[j][k].value = inputArr[j][k];
			}
		}
	}
	model=svm_train(&prob,&param);
	const char* model_file_name="C://Users//pc//Desktop//Model.txt";
	svm_save_model(model_file_name,model);
	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);
	for(int i=0;i<sample_count;i++)
	{
		delete prob.x[i];
	}
	delete[] prob.x;
	delete[] prob.y;
}

int svmPredict(const char*model)
{
	struct svm_node *test;
	struct svm_model *testmodel;
	testmodel=svm_load_model(model);
	int feature_dim=12;
	test=new struct svm_node[feature_dim+1];
	for(int i=0;i<feature_dim;i++)
	{
		test[i].index=i+1;
		test[i].value=testArr[i];
	}
	test[feature_dim].index=-1;
	double p=svm_predict(testmodel,test);
	svm_free_and_destroy_model(&testmodel);
	delete[] test;
	if(p>0.5)
	{
		return 1;
	}
	else{
		return -1;
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	svmTraining();
	int flag=svmPredict("C://Users//pc//Desktop//Model.txt");
	cout<<"flag="<<flag<<endl;
	system("pause");
	return 0;
}

