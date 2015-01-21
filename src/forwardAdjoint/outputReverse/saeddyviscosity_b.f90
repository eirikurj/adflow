   !        Generated by TAPENADE     (INRIA, Tropics team)
   !  Tapenade 3.10 (r5363) -  9 Sep 2014 09:53
   !
   !  Differentiation of saeddyviscosity in reverse (adjoint) mode (with options i4 dr8 r8 noISIZE):
   !   gradient     of useful results: *rev *w *rlv
   !   with respect to varying inputs: *rev *w *rlv
   !   Plus diff mem management of: rev:in w:in rlv:in
   !      ==================================================================
   !      ==================================================================
   SUBROUTINE SAEDDYVISCOSITY_B()
   !
   !      ******************************************************************
   !      *                                                                *
   !      * saEddyViscosity computes the eddy-viscosity according to the   *
   !      * Spalart-Allmaras model for the block given in blockPointers.   *
   !      * This routine for both the original version as well as the      *
   !      * modified version according to Edwards.                         *
   !      *                                                                *
   !      ******************************************************************
   !
   USE BLOCKPOINTERS
   USE CONSTANTS
   USE PARAMTURB
   IMPLICIT NONE
   !
   !      Local variables.
   !
   INTEGER(kind=inttype) :: i, j, k
   REAL(kind=realtype) :: chi, chi3, fv1, rnusa, cv13
   REAL(kind=realtype) :: chid, chi3d, fv1d, rnusad
   REAL(kind=realtype) :: tempd
   REAL(kind=realtype) :: tempd0
   !
   !      ******************************************************************
   !      *                                                                *
   !      * Begin execution                                                *
   !      *                                                                *
   !      ******************************************************************
   !
   ! Store the cv1^3; cv1 is a constant of the Spalart-Allmaras model.
   cv13 = rsacv1**3
   DO k=ke,1,-1
   DO j=je,1,-1
   DO i=ie,1,-1
   rnusa = w(i, j, k, itu1)*w(i, j, k, irho)
   chi = rnusa/rlv(i, j, k)
   chi3 = chi**3
   fv1 = chi3/(chi3+cv13)
   fv1d = rnusa*revd(i, j, k)
   tempd0 = fv1d/(cv13+chi3)
   chi3d = (1.0_8-chi3/(cv13+chi3))*tempd0
   chid = 3*chi**2*chi3d
   tempd = chid/rlv(i, j, k)
   rnusad = tempd + fv1*revd(i, j, k)
   revd(i, j, k) = 0.0_8
   rlvd(i, j, k) = rlvd(i, j, k) - rnusa*tempd/rlv(i, j, k)
   wd(i, j, k, itu1) = wd(i, j, k, itu1) + w(i, j, k, irho)*rnusad
   wd(i, j, k, irho) = wd(i, j, k, irho) + w(i, j, k, itu1)*rnusad
   END DO
   END DO
   END DO
   END SUBROUTINE SAEDDYVISCOSITY_B
