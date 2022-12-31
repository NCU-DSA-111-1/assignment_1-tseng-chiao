# include <stdio.h>
# include "func.h"
# include <time.h>
# include <stdlib.h>
# include <math.h>
# define rando() ((double)rand()/((double)RAND_MAX+1))
void print_test_matrix(int (*)[5], double (*)[2],int NumPattern, int NumInput,int NumOutput);
/*show the testing result*/ 

int training_data_generation(int (*Input)[3], int (*Target)[2]){     //random generation of the train data
    srand(time(0));
    printf("training data:\n");
    int add=0;
    for(int i=0;i<5;i++){
        Input[i][0]=0;
    }
    for(int j=1;j<2;j++){
        Input[1][j]=0;
    }
    for(int i=1;i<5;i++){
        for(int j=1; j<3;j++){
            Input[i][j]=rand()%2;           
        }
    }
    for(int i=0;i<5;i++){
        for(int j=0; j<3;j++){
            printf("%d ,",Input[i][j]);
        }
        printf("\n");
    }
    printf("target data:");
    for(int i=1;i<5;i++){
    add= (Input[i][1]+ Input[i][2])%2;
    if(add==0){
        Target[i][1]=0;
    }
    if(add==1){
        Target[i][1]=1;
    }
     for(int j=0;j<2;j++){
        Target[0][j]=0;
     }
    for(int i=1;i<5;i++){
        Target[i][0]=0;
    }
    }
   
    printf("\n");
    for(int i=0;i<5;i++){
        for(int j=0; j<2;j++){
            printf("%d ,",Target[i][j]);
        }
        printf("\n");
    }

    return 0;
}

void weight_initialize(double (*WeightIH)[3],double (*DeltaWeightIH)[3],double (*WeightHO)[3],double (*DeltaWeightHO)[3],int NumHidden,int NumInput,double smallwt,int NumOutput){
    for(int j = 1 ; j <= NumHidden ; j++ ) {    /* initialize WeightIH and DeltaWeightIH */
        for( int i = 0 ; i <= NumInput ; i++ ) { 
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
    for(int k = 1 ; k <= NumOutput ; k ++ ) {    /* initialize WeightHO and DeltaWeightHO */
        for(int j = 0 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;              
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
}

void print_train_matrix(int (*Input)[3],double (*Output)[2],int (*Target)[2], int NumPattern, int NumInput,int NumOutput){
for(int p = 1 ; p <= NumPattern ; p++ ) {          /*Numpattern: test numbers*/
    fprintf(stdout, "\n%d\t", p) ;
        for(int i = 1 ; i <= NumInput ; i++ ) {
            fprintf(stdout, "%d\t", Input[p][i]) ;
        }
        for(int k = 1 ; k <= NumOutput; k++ ) {
            fprintf(stdout, "%d\t%lf\t", Target[p][k], Output[p][k]) ;
        }
    }
}

void test_data(int NUMHID,int NUMIN,int NUMPAT,int NUMOUT,int NumPattern,int NumInput,int (*ranpat),int NumHidden,double (*SumH)[3],double (*WeightIH)[3],double (*Hidden)[3],int NumOutput,double (*SumO)[2],double (*WeightHO)[2]){
        int (*Input)[NUMIN+1]=calloc((NUMPAT+1), sizeof(*Input));
        double (*Output)[NUMOUT+1]=calloc((NUMPAT+1), sizeof(*Output));
        
        for(int i=1;i<=NumPattern;i++){
            for(int j=1;j<=NumInput;j++){
                scanf("%d",&Input[i][j]);
            }
        }
        for(int  np = 1 ; np <= NumPattern ; np++ ) {    /* repeat for all the training patterns */
            int p;
            ranpat[np]=p;
            for(int j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit activations */
                SumH[p][j] = WeightIH[0][j] ;
                for(int i = 1 ; i <= NumInput ; i++ ) {
                    SumH[p][j] += Input[p][i] * WeightIH[i][j] ;
                }
                Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
            }
            for( int k = 1 ; k <= NumOutput ; k++ ) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO[0][k] ;
                for(int j = 1 ; j <= NumHidden ; j++ ) {
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
                }
                Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs */
            }
        }
    free(SumH);
    free(SumO);
    free(Hidden);
    print_test_matrix(Input,Output,NumPattern,NumInput,NumOutput); 
    printf("\n");
}

void print_test_matrix(int (*Input)[5], double (*Output)[2],int NumPattern, int NumInput,int NumOutput){
    printf("\nPat\t"); 
    for(int i = 1 ; i <= NumInput ; i++ ) {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    printf("Output\t") ;
    for(int p = 1 ; p <= NumPattern ; p++ ) {          /*Numpattern: test numbers*/
    fprintf(stdout, "\n%d\t", p) ;
        for(int i = 1 ; i <= NumInput ; i++ ) {
            fprintf(stdout, "%d\t", Input[p][i]) ;
        }
        for(int k = 1 ; k <= NumOutput; k++ ) {
            fprintf(stdout, "%lf\t", Output[p][k]) ;
        }
    }
    free(Input);
    free(Output);
}



