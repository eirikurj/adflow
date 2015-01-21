   !        Generated by TAPENADE     (INRIA, Tropics team)
   !  Tapenade 3.10 (r5363) -  9 Sep 2014 09:53
   !
   !  Differentiation of setbcpointersbwd in reverse (adjoint) mode (with options i4 dr8 r8 noISIZE):
   !   gradient     of useful results: *rev *p *w *rlv rev1 rev2 pp1
   !                pp2 rlv1 rlv2 ww1 ww2
   !   with respect to varying inputs: *rev *p *w *rlv
   !   Plus diff mem management of: rev:in p:in w:in rlv:in
   !
   !      ******************************************************************
   !      *                                                                *
   !      * File:          setBcPointers.f90                               *
   !      * Author:        Edwin van der Weide                             *
   !      * Starting date: 02-17-2004                                      *
   !      * Last modified: 06-12-2005                                      *
   !      *                                                                *
   !      ******************************************************************
   !
   SUBROUTINE SETBCPOINTERSBWD_B(nn, ww1, ww1d, ww2, ww2d, pp1, pp1d, pp2, &
   & pp2d, rlv1, rlv1d, rlv2, rlv2d, rev1, rev1d, rev2, rev2d, offset)
   !
   !      ******************************************************************
   !      *                                                                *
   !      * setBCPointers sets the pointers needed for the boundary        *
   !      * condition treatment on a general face, such that the boundary  *
   !      * routines are only implemented once instead of 6 times.         *
   !      *                                                                *
   !      ******************************************************************
   !
   USE BCTYPES
   USE BLOCKPOINTERS
   USE FLOWVARREFSTATE
   IMPLICIT NONE
   !
   !      Subroutine arguments.
   !
   INTEGER(kind=inttype), INTENT(IN) :: nn, offset
   REAL(kind=realtype), DIMENSION(imaxdim, jmaxdim, nw) :: ww1, ww2
   REAL(kind=realtype), DIMENSION(imaxdim, jmaxdim, nw) :: ww1d, ww2d
   REAL(kind=realtype), DIMENSION(imaxdim, jmaxdim) :: pp1, pp2
   REAL(kind=realtype), DIMENSION(imaxdim, jmaxdim) :: pp1d, pp2d
   REAL(kind=realtype), DIMENSION(imaxdim, jmaxdim) :: rlv1, rlv2
   REAL(kind=realtype), DIMENSION(imaxdim, jmaxdim) :: rlv1d, rlv2d
   REAL(kind=realtype), DIMENSION(imaxdim, jmaxdim) :: rev1, rev2
   REAL(kind=realtype), DIMENSION(imaxdim, jmaxdim) :: rev1d, rev2d
   !
   !      Local variables
   !
   INTEGER(kind=inttype) :: id, ih, ierr, i, j, k
   INTEGER :: branch
   !
   !      ******************************************************************
   !      *                                                                *
   !      * Begin execution                                                *
   !      *                                                                *
   !      ******************************************************************
   !
   ! Determine the face id on which the subface is located and set
   ! the pointers accordinly.
   SELECT CASE  (bcfaceid(nn)) 
   CASE (imin) 
   id = 2 + offset
   ih = 1 - offset
   DO k=1,ke
   DO j=1,je
   IF (viscous) THEN
   CALL PUSHCONTROL1B(0)
   ELSE
   CALL PUSHCONTROL1B(1)
   END IF
   IF (eddymodel) THEN
   CALL PUSHCONTROL1B(1)
   ELSE
   CALL PUSHCONTROL1B(0)
   END IF
   END DO
   END DO
   DO k=ke,1,-1
   DO j=je,1,-1
   CALL POPCONTROL1B(branch)
   IF (branch .NE. 0) THEN
   revd(id, j, k) = revd(id, j, k) + rev2d(j, k)
   rev2d(j, k) = 0.0_8
   revd(ih, j, k) = revd(ih, j, k) + rev1d(j, k)
   rev1d(j, k) = 0.0_8
   END IF
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) THEN
   rlvd(id, j, k) = rlvd(id, j, k) + rlv2d(j, k)
   rlv2d(j, k) = 0.0_8
   rlvd(ih, j, k) = rlvd(ih, j, k) + rlv1d(j, k)
   rlv1d(j, k) = 0.0_8
   END IF
   pd(id, j, k) = pd(id, j, k) + pp2d(j, k)
   pp2d(j, k) = 0.0_8
   pd(ih, j, k) = pd(ih, j, k) + pp1d(j, k)
   pp1d(j, k) = 0.0_8
   wd(id, j, k, :) = wd(id, j, k, :) + ww2d(j, k, :)
   ww2d(j, k, :) = 0.0_8
   wd(ih, j, k, :) = wd(ih, j, k, :) + ww1d(j, k, :)
   ww1d(j, k, :) = 0.0_8
   END DO
   END DO
   CASE (imax) 
   !===============================================================
   id = il - offset
   ih = ie + offset
   DO k=1,ke
   DO j=1,je
   IF (viscous) THEN
   CALL PUSHCONTROL1B(0)
   ELSE
   CALL PUSHCONTROL1B(1)
   END IF
   IF (eddymodel) THEN
   CALL PUSHCONTROL1B(1)
   ELSE
   CALL PUSHCONTROL1B(0)
   END IF
   END DO
   END DO
   DO k=ke,1,-1
   DO j=je,1,-1
   CALL POPCONTROL1B(branch)
   IF (branch .NE. 0) THEN
   revd(id, j, k) = revd(id, j, k) + rev2d(j, k)
   rev2d(j, k) = 0.0_8
   revd(ih, j, k) = revd(ih, j, k) + rev1d(j, k)
   rev1d(j, k) = 0.0_8
   END IF
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) THEN
   rlvd(id, j, k) = rlvd(id, j, k) + rlv2d(j, k)
   rlv2d(j, k) = 0.0_8
   rlvd(ih, j, k) = rlvd(ih, j, k) + rlv1d(j, k)
   rlv1d(j, k) = 0.0_8
   END IF
   pd(id, j, k) = pd(id, j, k) + pp2d(j, k)
   pp2d(j, k) = 0.0_8
   pd(ih, j, k) = pd(ih, j, k) + pp1d(j, k)
   pp1d(j, k) = 0.0_8
   wd(id, j, k, :) = wd(id, j, k, :) + ww2d(j, k, :)
   ww2d(j, k, :) = 0.0_8
   wd(ih, j, k, :) = wd(ih, j, k, :) + ww1d(j, k, :)
   ww1d(j, k, :) = 0.0_8
   END DO
   END DO
   CASE (jmin) 
   !===============================================================
   id = 2 + offset
   ih = 1 - offset
   DO k=1,ke
   DO i=1,ie
   IF (viscous) THEN
   CALL PUSHCONTROL1B(0)
   ELSE
   CALL PUSHCONTROL1B(1)
   END IF
   IF (eddymodel) THEN
   CALL PUSHCONTROL1B(1)
   ELSE
   CALL PUSHCONTROL1B(0)
   END IF
   END DO
   END DO
   DO k=ke,1,-1
   DO i=ie,1,-1
   CALL POPCONTROL1B(branch)
   IF (branch .NE. 0) THEN
   revd(i, id, k) = revd(i, id, k) + rev2d(i, k)
   rev2d(i, k) = 0.0_8
   revd(i, ih, k) = revd(i, ih, k) + rev1d(i, k)
   rev1d(i, k) = 0.0_8
   END IF
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) THEN
   rlvd(i, id, k) = rlvd(i, id, k) + rlv2d(i, k)
   rlv2d(i, k) = 0.0_8
   rlvd(i, ih, k) = rlvd(i, ih, k) + rlv1d(i, k)
   rlv1d(i, k) = 0.0_8
   END IF
   pd(i, id, k) = pd(i, id, k) + pp2d(i, k)
   pp2d(i, k) = 0.0_8
   pd(i, ih, k) = pd(i, ih, k) + pp1d(i, k)
   pp1d(i, k) = 0.0_8
   wd(i, id, k, :) = wd(i, id, k, :) + ww2d(i, k, :)
   ww2d(i, k, :) = 0.0_8
   wd(i, ih, k, :) = wd(i, ih, k, :) + ww1d(i, k, :)
   ww1d(i, k, :) = 0.0_8
   END DO
   END DO
   CASE (jmax) 
   !===============================================================
   id = jl - offset
   ih = je + offset
   DO k=1,ke
   DO i=1,ie
   IF (viscous) THEN
   CALL PUSHCONTROL1B(0)
   ELSE
   CALL PUSHCONTROL1B(1)
   END IF
   IF (eddymodel) THEN
   CALL PUSHCONTROL1B(1)
   ELSE
   CALL PUSHCONTROL1B(0)
   END IF
   END DO
   END DO
   DO k=ke,1,-1
   DO i=ie,1,-1
   CALL POPCONTROL1B(branch)
   IF (branch .NE. 0) THEN
   revd(i, id, k) = revd(i, id, k) + rev2d(i, k)
   rev2d(i, k) = 0.0_8
   revd(i, ih, k) = revd(i, ih, k) + rev1d(i, k)
   rev1d(i, k) = 0.0_8
   END IF
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) THEN
   rlvd(i, id, k) = rlvd(i, id, k) + rlv2d(i, k)
   rlv2d(i, k) = 0.0_8
   rlvd(i, ih, k) = rlvd(i, ih, k) + rlv1d(i, k)
   rlv1d(i, k) = 0.0_8
   END IF
   pd(i, id, k) = pd(i, id, k) + pp2d(i, k)
   pp2d(i, k) = 0.0_8
   pd(i, ih, k) = pd(i, ih, k) + pp1d(i, k)
   pp1d(i, k) = 0.0_8
   wd(i, id, k, :) = wd(i, id, k, :) + ww2d(i, k, :)
   ww2d(i, k, :) = 0.0_8
   wd(i, ih, k, :) = wd(i, ih, k, :) + ww1d(i, k, :)
   ww1d(i, k, :) = 0.0_8
   END DO
   END DO
   CASE (kmin) 
   !===============================================================
   id = 2 + offset
   ih = 1 - offset
   DO j=1,je
   DO i=1,ie
   IF (viscous) THEN
   CALL PUSHCONTROL1B(0)
   ELSE
   CALL PUSHCONTROL1B(1)
   END IF
   IF (eddymodel) THEN
   CALL PUSHCONTROL1B(1)
   ELSE
   CALL PUSHCONTROL1B(0)
   END IF
   END DO
   END DO
   DO j=je,1,-1
   DO i=ie,1,-1
   CALL POPCONTROL1B(branch)
   IF (branch .NE. 0) THEN
   revd(i, j, id) = revd(i, j, id) + rev2d(i, j)
   rev2d(i, j) = 0.0_8
   revd(i, j, ih) = revd(i, j, ih) + rev1d(i, j)
   rev1d(i, j) = 0.0_8
   END IF
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) THEN
   rlvd(i, j, id) = rlvd(i, j, id) + rlv2d(i, j)
   rlv2d(i, j) = 0.0_8
   rlvd(i, j, ih) = rlvd(i, j, ih) + rlv1d(i, j)
   rlv1d(i, j) = 0.0_8
   END IF
   pd(i, j, id) = pd(i, j, id) + pp2d(i, j)
   pp2d(i, j) = 0.0_8
   pd(i, j, ih) = pd(i, j, ih) + pp1d(i, j)
   pp1d(i, j) = 0.0_8
   wd(i, j, id, :) = wd(i, j, id, :) + ww2d(i, j, :)
   ww2d(i, j, :) = 0.0_8
   wd(i, j, ih, :) = wd(i, j, ih, :) + ww1d(i, j, :)
   ww1d(i, j, :) = 0.0_8
   END DO
   END DO
   CASE (kmax) 
   !===============================================================
   id = kl - offset
   ih = ke + offset
   DO j=1,je
   DO i=1,ie
   IF (viscous) THEN
   CALL PUSHCONTROL1B(0)
   ELSE
   CALL PUSHCONTROL1B(1)
   END IF
   IF (eddymodel) THEN
   CALL PUSHCONTROL1B(1)
   ELSE
   CALL PUSHCONTROL1B(0)
   END IF
   END DO
   END DO
   DO j=je,1,-1
   DO i=ie,1,-1
   CALL POPCONTROL1B(branch)
   IF (branch .NE. 0) THEN
   revd(i, j, id) = revd(i, j, id) + rev2d(i, j)
   rev2d(i, j) = 0.0_8
   revd(i, j, ih) = revd(i, j, ih) + rev1d(i, j)
   rev1d(i, j) = 0.0_8
   END IF
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) THEN
   rlvd(i, j, id) = rlvd(i, j, id) + rlv2d(i, j)
   rlv2d(i, j) = 0.0_8
   rlvd(i, j, ih) = rlvd(i, j, ih) + rlv1d(i, j)
   rlv1d(i, j) = 0.0_8
   END IF
   pd(i, j, id) = pd(i, j, id) + pp2d(i, j)
   pp2d(i, j) = 0.0_8
   pd(i, j, ih) = pd(i, j, ih) + pp1d(i, j)
   pp1d(i, j) = 0.0_8
   wd(i, j, id, :) = wd(i, j, id, :) + ww2d(i, j, :)
   ww2d(i, j, :) = 0.0_8
   wd(i, j, ih, :) = wd(i, j, ih, :) + ww1d(i, j, :)
   ww1d(i, j, :) = 0.0_8
   END DO
   END DO
   END SELECT
   END SUBROUTINE SETBCPOINTERSBWD_B
