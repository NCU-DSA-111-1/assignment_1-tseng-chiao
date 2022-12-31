# ifndef __FUNC_H__
# define __FUNC_H__

int training_data_generation(int (*)[3], int (*)[2]);  
/*random generation of the train data*/
void weight_initialize(double (*)[3],double (*)[3],double (*)[3],double (*)[3],int NumHidden,int NumInput,double smallwt,int NumOutput);
/*give random weighting*/
void print_train_matrix(int (*)[3],double (*)[2],int (*)[2], int NumPattern, int NumInput,int NumOutput);    
/*show the training result*/
void test_data(int NUMHID,int NUMIN,int NUMPAT,int NUMOUT,int NumPattern,int NumInput,int *,int NumHidden,double (*)[3],double (*)[3],double (*)[3],int NumOutput,double (*)[2],double (*)[2]);   
/*use the computed weighting to compute the testing data*/

# endif