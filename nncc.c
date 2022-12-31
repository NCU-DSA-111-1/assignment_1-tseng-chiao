#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include "func.h"

#define NUMPAT 4    /*input 4 pairs train data*/
#define NUMIN  2    /*input 2-bits data*/
#define NUMHID 2    
#define NUMOUT 1

#define rando() ((double)rand()/((double)RAND_MAX+1))   /*generate random weight*/
int main() {
    int    i, j, k, p, np, op, ranpat[NUMPAT+1], epoch;
    int    NumPattern = NUMPAT, NumInput = NUMIN, NumHidden = NUMHID, NumOutput = NUMOUT;  
    double (*SumH)[NUMHID+1]=calloc((NUMPAT+1), sizeof(*SumH));
    double (*WeightIH)[NUMHID+1]=calloc((NUMIN+1), sizeof(*WeightIH)); 
    double (*Hidden)[NUMHID+1]=calloc((NUMPAT+1), sizeof(*Hidden));
    double (*SumO)[NUMOUT+1]=calloc((NUMPAT+1), sizeof(*SumO)); 
    double (*WeightHO)[NUMOUT+1]=calloc((NUMHID+1), sizeof(*WeightHO)); 
    double (*Output)[NUMOUT+1]=calloc((NUMPAT+1), sizeof(*Output));
    double (*DeltaO)[NUMOUT+1]=malloc(sizeof(*DeltaO));
    double (*SumDOW)[NUMHID+1]=malloc(sizeof(*SumDOW)), (*DeltaH)[NUMHID+1]=malloc(sizeof(*DeltaH));
    double (*DeltaWeightIH)[NUMHID+1]=calloc((NUMIN+1), sizeof(*DeltaWeightIH)); 
    double (*DeltaWeightHO)[NUMOUT+1]=calloc((NUMHID+1), sizeof(*DeltaWeightHO));
    double Error, eta = 0.5, alpha = 0.9, smallwt = 0.5;    /*initialize training rate  parameters, will replace in the following loop*/
    int (*Input)[3]=calloc(5,sizeof(*Input));
    int (*Target)[2]=calloc(5,sizeof(*Target));
    training_data_generation(Input, Target);/**/
    double (*w)[3]=malloc(sizeof(*w));
    double (*y)[3]=malloc(sizeof(*y));
    double (*z)[3]=malloc(sizeof(*z));
    double (*x)[3]=malloc(sizeof(*x));
    weight_initialize(w,x,y,z,NumHidden,NumInput,smallwt,NumOutput);/**/
     
    for( epoch = 0 ; epoch < 100000 ; epoch++) {    /* iterate weight updates */
        for( p = 1 ; p <= NumPattern ; p++ ) {    /* randomize order of individuals */
            ranpat[p] = p ;
        }
        for( p = 1 ; p <= NumPattern ; p++) {
            np = p + rando() * ( NumPattern + 1 - p ) ;
            op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        }
        Error = 0.0 ;
        for( np = 1 ; np <= NumPattern ; np++ ) {    /* repeat for all the training patterns */
            p = ranpat[np];
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit activations */
                SumH[p][j] = WeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) {
                    SumH[p][j] += Input[p][i] * WeightIH[i][j] ;
                }
                Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
            }
            for( k = 1 ; k <= NumOutput ; k++ ) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
                }
                Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs */
                Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]) ;   /* SSE */
                (*DeltaO)[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   /* Sigmoidal Outputs, SSE */
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                (*SumDOW)[j] = 0.0 ;
                for( k = 1 ; k <= NumOutput ; k++ ) {
                    (*SumDOW)[j] += WeightHO[j][k] * (*DeltaO)[k] ;
                }
                (*DeltaH)[j] = (*SumDOW)[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {     /* update weights WeightIH */
                DeltaWeightIH[0][j] = eta * (*DeltaH)[j] + alpha * DeltaWeightIH[0][j] ;
                WeightIH[0][j] += DeltaWeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) { 
                    DeltaWeightIH[i][j] = eta * Input[p][i] * (*DeltaH)[j] + alpha * DeltaWeightIH[i][j];
                    WeightIH[i][j] += DeltaWeightIH[i][j] ;
                }
            }
            for( k = 1 ; k <= NumOutput ; k ++ ) {    /* update weights WeightHO */
                DeltaWeightHO[0][k] = eta * (*DeltaO)[k] + alpha * DeltaWeightHO[0][k] ;
                WeightHO[0][k] += DeltaWeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    DeltaWeightHO[j][k] = eta * Hidden[p][j] * (*DeltaO)[k] + alpha * DeltaWeightHO[j][k] ;
                    WeightHO[j][k] += DeltaWeightHO[j][k] ;
                }
            }
        }
        if( epoch%100 == 0 ) fprintf(stdout, "\nEpoch %-5d :   Error = %f", epoch, Error) ; /*print error per 100 times*/
        if( Error < 0.0004 ) break ;  /* stop learning when 'near enough' */
    }
    
    fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d\n\nPat\t", epoch) ;   /* print network outputs */
    for( i = 1 ; i <= NumInput ; i++ ) {
        fprintf(stdout, "Input%-2d\t", i) ;
    }
    for( k = 1 ; k <= NumOutput ; k++ ) {
        fprintf(stdout, "Target%d\tOutput%d\t", k, k) ;
    }
    free(SumH);
    free(SumO);
    free(Hidden);
    free(DeltaO);
    free(SumDOW);
    free(DeltaWeightIH);
    free(DeltaWeightHO);
    free(w);    
    free(y);

    print_train_matrix(Input,Output,Target,NumPattern,NumInput,NumOutput);/**/
    free(Input);
    free(Output);
    free(Target);
    fprintf(stdout, "\n\nIt's your turn!\nPlease enter 8 numbers composed of 0 or 1(tab needed)\nSorry! I can't do the testing data now!\n\n") ;

    test_data(NUMHID,NUMIN,NUMPAT,NUMOUT,NumPattern,NumInput,ranpat,NumHidden,SumH,WeightIH, Hidden,NumOutput,SumO,WeightHO);
    return 1 ;
}
