#include <Python.h>
#include "numpy/arrayobject.h"
#include <stdlib.h>
#if defined (_OPENMP)
#  include <omp.h>
#endif
/*
This file contains a C implementation of the calc_OFTI method and submethods:
- eccentricity_anomaly
- mikkola_solve
- danbySolve
- calc_accel
- scale_and_rotate
- calc_XYZ
- calc_OFTI

Only the calc_OFTI method can interface with a python program,
the other are internal helper functions based on the original python version.

This C program provides OpenMP support.

The only difference in interfacing with the calc_OFTI program is that it will return 
one single numpy array containing all necessary variables versus a tuple.

*/
//to emulate the return tuple of python, fourDouble is a structure of four doubles
typedef struct fourDouble{
    double one,two,three,four;
}four_double;
//Calculates eccentricity anomaly -- C version
double eccentricity_anomaly(double E,double e,double M){
    return E-(e*sin(E))-M;
}
//Mikkola solver -- C version
double mikkola_solve(double M, double e){
    double alpha = (1.0-e)/((4.*e)+0.5);
    double beta = (0.5*M)/((4.*e)+0.5);
    double ab = sqrtf(pow(beta,2.0f)+pow(alpha,3.0f));
    double z = pow(copysign(1.0,beta+ab),1./3.);

    double s1 = z-alpha/z;
    double ds = -0.078*(pow(s1,5.0))/(1+e);
    double s = s1+ds;
    double E0 = M + e*(3.0*s-4.0*pow(s,3.0));
    double sinE = sin(E0);
    double cosE = cos(E0);
    double f = E0-e*sinE-M;
    double fp = 1. -e*cosE;
    double fpp = e*sinE;
    double fppp = e*cosE;
    double fpppp = -fpp;
    double dx1 = -f/fp;
    double dx2 = -f/(fp+0.5*fpp*dx1);
    double dx3 = -f/(fp+0.5*fpp*dx2 + (1./6.)*fppp*(dx2*dx2));
    double dx4 = -f/(fp+0.5*fpp*dx3 + (1./6.)*fppp*(dx3*dx3)+(1./24.)*(fpppp)*pow(dx3,3.0));
    return E0+dx4;
}
//Newton-Ralphson eccentricity anomaly solver ("Danby" method). C version
double danbySolve(double M0,double e,double h){
    int maxnum = 50;
    double k = 0.85;
    double E0 = M0 + copysign(1.0,sin(M0))*k*e;
    double lastE = E0;
    double nextE = lastE + 10.0*h;
    int number = 0;
    double delta_D = 1.;
    while((delta_D>h)&& number<maxnum+1){
        double ex = eccentricity_anomaly(nextE,e,M0);
        double ep = (1.0-e*cos(lastE));
        double epp = e*sin(lastE);
        double eppp = e*cos(lastE);
        lastE = nextE;
        double delta_N = -ex/ep;
        double delta_H = -ex/(ep+0.5*epp*delta_N);
        double delta_D = -ex/(ep+0.5*epp*delta_H+(1./6.)*eppp*delta_H*delta_H);
        nextE = lastE + delta_D;
        number+=1;
        if(number>=maxnum){
            nextE = mikkola_solve(M0,e);
        }
    }
    return nextE;
}
//calculate acceleration --  C version
four_double *calc_accel(double a,double T,double to,double e,double i,double w,double O,double date,double dist, double E){
    double a_km=a*dist*149598073.0;
    double n = (2.0*M_PI)/T;
    double f1=sqrt(1.0+e)*sin(E/2.0);
    double f2 = sqrt(1.0-e)*cos(E/2.0);
    double f = 2.*atan2(f1,f2);
    double r=(a_km*(1-e*e))/(1+e*cos(f));
    double Edot = n/(1-e*cos(E));
    double rdot = e*sin(f)*((n*a_km)/(sqrt(1-e*e)));
    double fdot = (n*(1+e*cos(f)))/(1-e*e)*sin(f)/sin(E);
    double Eddot = ((-n*e*sin(f))/(1-e*e))*fdot;
    double rddot = a_km*e*cos(E)*(Edot*Edot)+a_km*e*sin(E)*Eddot;
    double fddot = Eddot*(sin(f)/sin(E))-(Edot*Edot)*(e*sin(f)/(1-e*cos(E)));
    double Xddot = (rddot - r*fdot*fdot)*(cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i)) + 
            (-2*rdot*fdot - r*fddot)*(cos(O)*sin(w+f) + sin(O)*cos(w+f)*cos(i));
    double Yddot = (rddot - r*fdot*fdot)*(sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i)) + 
            (2*rdot*fdot + r*fddot)*(sin(O)*sin(w+f) + cos(O)*cos(w+f)*cos(i));
    double Zddot = sin(i)*((rddot - r*(fdot*fdot))*sin(w+f) + ((2*rdot*fdot + r*fddot)*cos(w+f)));
    four_double* return_tuple = (four_double*)malloc(sizeof(four_double));
    double constantMultiply = 1000./(60.*60.*24.*365.);
    return_tuple -> one = Xddot*constantMultiply;
    return_tuple -> two = Yddot*constantMultiply;
    return_tuple -> three = Zddot*constantMultiply;
    return return_tuple;
}

double rand_gen() {
   return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}
//scale and rotate --  C version
four_double *scale_and_rotate(double xi, double yi, double rho1, double rho2, double pa1, double pa2, double ai, double constantM,double m1,double dist,double d){
    double PA_rand = sqrt(-2*log((double)rand() / (double)RAND_MAX))*cos(2*M_PI*(double)rand() / (double)RAND_MAX)*pa2+pa1;
        double r_model = sqrt(xi*xi+yi*yi);
        double rho_rand = sqrt(-2*log(rand_gen()))*cos(2*M_PI*rand_gen())*rho2/1000.+rho1/1000.;
        double a2i = ai*(rho_rand/r_model);
        double a2_au = a2i*dist;
        double T2i = sqrt(pow(fabs(a2_au),3.0)/fabs(m1));
        double to2i = d-(constantM*T2i);
        double PA_model = fmod((atan2(xi,-yi)*180/M_PI+270),360.0);
        double O2i;
        if(PA_model<0){
            O2i = PA_rand-PA_model+360.0;
        }
        else{
            O2i = PA_rand-PA_model;
        }
        O2i*=M_PI/180;
        
           

    four_double* return_tuple = (four_double*)malloc(sizeof(four_double));
    return_tuple->one=a2i;
    return_tuple->two=T2i;
    return_tuple->three=to2i;
    return_tuple->four=O2i;

    return return_tuple;
}
//calculate XYZ --  C version
four_double* calc_XYZ(double a,double T,double to,double e,double i,double w,double O,double date) {
        double M0 = (2*M_PI)/T*(date-to);
        double eTemp = e;
        double ETemp;
        ETemp = danbySolve(M0,eTemp,0.001);
        double f1 = sqrt(1.+eTemp)*sin(ETemp/2.);
        double f2 = sqrt(1.-eTemp)*cos(ETemp/2.0);
        double f = 2.0*atan2(f1,f2);
        double r = a*(1.0-eTemp*eTemp)/(1.0+(eTemp*cos(f)));

        double xAdd = r*(cos(O)*cos(w+f)-sin(O)*sin(w+f)*cos(i));
        double yAdd = r*(sin(O)*cos(w+f)+cos(O)*sin(w+f)*cos(i));
        double zAdd = r*sin(w+f)*sin(i);
        four_double* return_tuple = (four_double*)malloc(sizeof(four_double));
        return_tuple->one=xAdd;
        return_tuple->two=yAdd;
        return_tuple->three=zAdd;
        return_tuple->four=ETemp;

        return return_tuple;
}
//calculate velocities --  C version
four_double* calc_velocities(double a, double T, double to, double e, double i, double w, double O, double date, double dist,double E){
    double a_km = a*dist*149598073.0;
    double n = (2*M_PI)/T;
    double f1 = sqrt(1.+e)*sin(E/2.);
    double f2 = sqrt(1.-e)*cos(E/2.);
    double f = 2.*atan2(f1,f2);
    double rdot = ((n*a_km)/(sqrt(1.-e*e)))*e*sin(f);
    double rfdot = ((n*a_km)/sqrt(1.-e*e))*(1.+e*cos(f));
    double Xdot = rdot*(cos(O)*cos(w+f)-sin(O)*sin(w+f)*cos(i))+rfdot*(-cos(O)*sin(w+f)-sin(O)*cos(w+f)*cos(i));
    double Ydot = rdot * (sin(O)*cos(w+f)+cos(O)*sin(w+f)*cos(i))+rfdot*(-sin(O)*sin(w+f)+cos(O)*cos(w+f)*cos(i));
    double Zdot = ((n*a_km)/(sqrt(1-e*e)))*sin(i)*(cos(w+f)+e*cos(w));
    four_double* return_tuple = (four_double*)malloc(sizeof(four_double));
    double constantMultiply = 1./(60.*60.*24.*365.);
    return_tuple-> one = Xdot*constantMultiply;
    return_tuple -> two = Ydot*constantMultiply;
    return_tuple-> three = Zdot*constantMultiply;
    return return_tuple;

}
PyObject* calcOFTI(PyObject *self, PyObject *args){
    double* a,*constant,*T,*to,*e,*i,*w,*O,*m1,*dist;
    double date;
    PyObject *rho,*pa;
    PyObject* ArrayTotalObject;
    if(!PyArg_ParseTuple(args, "OdOO",&ArrayTotalObject,&date,&rho,&pa)) {
        return NULL;
    }
    //convert to contiguous array so data can be accessed easily
    PyArrayObject* ArrayTotal = (PyArrayObject *)PyArray_ContiguousFromObject(ArrayTotalObject,NPY_FLOAT64,1,2);
    double* arrayData = (double*)(ArrayTotal->data);
    int m = ArrayTotal ->dimensions[1];
    //set variables as subarrays of main array
    a = &arrayData[0];
    T = &arrayData[1*m];
    constant = &arrayData[2*m];
    to = &arrayData[3*m];
    e = &arrayData[4*m];
    i = &arrayData[5*m];
    w = &arrayData[6*m];
    O = &arrayData[7*m];
    m1 = &arrayData[8*m];
    dist = &arrayData[9*m];
    //extract values from input tuples
    double rho1 = PyFloat_AsDouble(PyTuple_GetItem(rho,0));
    double rho2 = PyFloat_AsDouble(PyTuple_GetItem(rho,1));
    double pa1 = PyFloat_AsDouble(PyTuple_GetItem(pa,0));
    double pa2 = PyFloat_AsDouble(PyTuple_GetItem(pa,1));
    int returnD1 = 19;
    //allocate return array
    double* dataReturn = (double*)malloc((size_t)m*returnD1*sizeof(double));
    #if defined (_OPENMP)
	#pragma omp parallel for
	#endif
    for(int j = 0; j< m;j++){
        four_double* calc_XYZReturn1 = calc_XYZ(a[j],T[j],to[j],e[j],i[j],w[j],O[j],date);
        four_double* sarR = scale_and_rotate(calc_XYZReturn1->one,calc_XYZReturn1->two,rho1,rho2,pa1,pa2,a[j],constant[j],m1[j],dist[j],date);
        four_double* calc_XYZReturn2 = calc_XYZ(sarR->one,sarR->two,sarR->three,e[j],i[j],w[j],sarR->four,date);
        calc_XYZReturn2->one=(calc_XYZReturn2->one)*1000.;
        calc_XYZReturn2->two = (calc_XYZReturn2->two)*1000.;
        calc_XYZReturn2->three = (calc_XYZReturn2->three)*1000.;
        four_double* calcVelocitiesR = calc_velocities(sarR->one,sarR->two,sarR->three,e[j],i[j],w[j],sarR->four,date,dist[j],calc_XYZReturn2->four);
        four_double* calcAccelR = calc_accel(sarR->one,sarR->two,sarR->three,e[j],i[j],w[j],sarR->four,date,dist[j],calc_XYZReturn2->four);
        //set values of return array
        dataReturn[0*m+j] = calc_XYZReturn2->one; // X2
        dataReturn[1*m+j] = calc_XYZReturn2->two; // Y2
        dataReturn[2*m+j] = calc_XYZReturn2->three; // Z2
        dataReturn[3*m+j] = calcVelocitiesR->one; // Xdot
        dataReturn[4*m+j]= calcVelocitiesR->two; // Ydot
        dataReturn[5*m+j] = calcVelocitiesR->three; // Zdot
        dataReturn[6*m+j] = calcAccelR->one; // Xddot
        dataReturn[7*m+j] = calcAccelR->two; // Yddot
        dataReturn[8*m+j] = calcAccelR->three; // Zddot
        dataReturn[9*m+j] = sarR->one; // a2
        dataReturn[10*m+j] = sarR->two; // T2
        dataReturn[11*m+j] = constant[j];
        dataReturn[12*m+j] = sarR->three; // to2
        dataReturn[13*m+j] = e[j]; // e
        dataReturn[14*m+j] = i[j]*180/M_PI; // i in degrees
        dataReturn[15*m+j] = w[j]*180/M_PI; // w in degrees
        dataReturn[16*m+j]= (sarR->four)*180/M_PI; // O2 in degrees
        dataReturn[17*m+j] = m1[j]; // total mass
        dataReturn[18*m+j] = dist[j]; // distance
        //free four_double structs
        free(calc_XYZReturn1);
        free(calc_XYZReturn2);
        free(sarR);
        free(calcVelocitiesR);
        free(calcAccelR);
    }
    
    //construct numpy array from double array to return
    npy_intp* dimArray2 = malloc((size_t)2*sizeof(npy_intp));
    dimArray2[0] = (npy_intp)returnD1;
    dimArray2[1] = (npy_intp)m;
    PyObject *returnMisc = PyArray_SimpleNewFromData(2,dimArray2,NPY_FLOAT64,(void*)dataReturn);
    free(dimArray2);
    return PyArray_Return((PyArrayObject *)returnMisc);



}
static char calcOFTI_Cdoc[] = " C version of the calc_OFTI method."
"If this fails, use python version with keyword python_version=True in the FitOrbit() call"
"Also contains internal calc_XYZ, calc_velocity,calc_accel,and rotate_and_scale methods"
"that cannot be interfaced with python";
static PyMethodDef calcOFTI_methods[] = {
    {"calcOFTI_C", calcOFTI, METH_VARARGS, calcOFTI_Cdoc},
    {NULL}
};


static struct PyModuleDef cFunctions = {
    PyModuleDef_HEAD_INIT,
    "cFunctions",
    calcOFTI_Cdoc,
    -1,
    calcOFTI_methods
};
PyMODINIT_FUNC PyInit_cFunctions(void) {
    PyObject* module = PyModule_Create(&cFunctions);
    import_array();
    return module;
}
