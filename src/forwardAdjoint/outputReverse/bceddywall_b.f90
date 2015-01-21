   !        Generated by TAPENADE     (INRIA, Tropics team)
   !  Tapenade 3.10 (r5363) -  9 Sep 2014 09:53
   !
   !  Differentiation of bceddywall in reverse (adjoint) mode (with options i4 dr8 r8 noISIZE):
   !   gradient     of useful results: *rev
   !   with respect to varying inputs: *rev
   !   Plus diff mem management of: rev:in bcdata:in
   !
   !      ******************************************************************
   !      *                                                                *
   !      * File:          bcEddyWall.f90                                  *
   !      * Author:        Georgi Kalitzin, Edwin van der Weide            *
   !      * Starting date: 06-11-2003                                      *
   !      * Last modified: 04-11-2005                                      *
   !      *                                                                *
   !      ******************************************************************
   !
   SUBROUTINE BCEDDYWALL_B(nn)
   !
   !      ******************************************************************
   !      *                                                                *
   !      * bcEddyWall sets the eddy viscosity in the halo cells of        *
   !      * viscous subface nn of the block given in blockPointers.        *
   !      * As the eddy viscosity is zero at the wall, the value in the    *
   !      * halo is simply the negative value of the first interior cell.  *
   !      *                                                                *
   !      ******************************************************************
   !
   USE BLOCKPOINTERS
   USE BCTYPES
   IMPLICIT NONE
   !
   !      Subroutine arguments.
   !
   INTEGER(kind=inttype), INTENT(IN) :: nn
   !
   !      Local variables.
   !
   INTEGER(kind=inttype) :: i, j
   REAL(kind=realtype) :: tmp
   REAL(kind=realtype) :: tmp0
   REAL(kind=realtype) :: tmp1
   REAL(kind=realtype) :: tmpd
   REAL(kind=realtype) :: tmpd1
   REAL(kind=realtype) :: tmpd0
   !
   !      ******************************************************************
   !      *                                                                *
   !      * Begin execution                                                *
   !      *                                                                *
   !      ******************************************************************
   !
   ! Determine the face id on which the subface is located and
   ! loop over the faces of the subface and set the eddy viscosity
   ! in the halo cells.
   SELECT CASE  (bcfaceid(nn)) 
   CASE (imin) 
   DO j=bcdata(nn)%jcend,bcdata(nn)%jcbeg,-1
   DO i=bcdata(nn)%icend,bcdata(nn)%icbeg,-1
   revd(2, i, j) = revd(2, i, j) - revd(1, i, j)
   revd(1, i, j) = 0.0_8
   END DO
   END DO
   CASE (imax) 
   DO j=bcdata(nn)%jcend,bcdata(nn)%jcbeg,-1
   DO i=bcdata(nn)%icend,bcdata(nn)%icbeg,-1
   tmpd = revd(ie, i, j)
   revd(ie, i, j) = 0.0_8
   revd(il, i, j) = revd(il, i, j) - tmpd
   END DO
   END DO
   CASE (jmin) 
   DO j=bcdata(nn)%jcend,bcdata(nn)%jcbeg,-1
   DO i=bcdata(nn)%icend,bcdata(nn)%icbeg,-1
   revd(i, 2, j) = revd(i, 2, j) - revd(i, 1, j)
   revd(i, 1, j) = 0.0_8
   END DO
   END DO
   CASE (jmax) 
   DO j=bcdata(nn)%jcend,bcdata(nn)%jcbeg,-1
   DO i=bcdata(nn)%icend,bcdata(nn)%icbeg,-1
   tmpd0 = revd(i, je, j)
   revd(i, je, j) = 0.0_8
   revd(i, jl, j) = revd(i, jl, j) - tmpd0
   END DO
   END DO
   CASE (kmin) 
   DO j=bcdata(nn)%jcend,bcdata(nn)%jcbeg,-1
   DO i=bcdata(nn)%icend,bcdata(nn)%icbeg,-1
   revd(i, j, 2) = revd(i, j, 2) - revd(i, j, 1)
   revd(i, j, 1) = 0.0_8
   END DO
   END DO
   CASE (kmax) 
   DO j=bcdata(nn)%jcend,bcdata(nn)%jcbeg,-1
   DO i=bcdata(nn)%icend,bcdata(nn)%icbeg,-1
   tmpd1 = revd(i, j, ke)
   revd(i, j, ke) = 0.0_8
   revd(i, j, kl) = revd(i, j, kl) - tmpd1
   END DO
   END DO
   END SELECT
   END SUBROUTINE BCEDDYWALL_B
