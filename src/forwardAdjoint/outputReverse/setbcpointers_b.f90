   !        Generated by TAPENADE     (INRIA, Tropics team)
   !  Tapenade 3.10 (r5363) -  9 Sep 2014 09:53
   !
   !  Differentiation of setbcpointers in reverse (adjoint) mode (with options i4 dr8 r8 noISIZE):
   !   Plus diff mem management of: p:in w:in rlv:in pp1:in-out pp2:in-out
   !                rlv1:in-out rlv2:in-out ww1:in-out ww2:in-out
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
   SUBROUTINE SETBCPOINTERS_B(nn, ww1, ww1b, ww2, ww2b, pp1, pp1b, pp2, &
   & pp2b, rlv1, rlv1b, rlv2, rlv2b, rev1, rev2, offset)
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
   USE BLOCKPOINTERS_B
   USE FLOWVARREFSTATE
   IMPLICIT NONE
   !
   !      Subroutine arguments.
   !
   INTEGER(kind=inttype), INTENT(IN) :: nn, offset
   REAL(kind=realtype), DIMENSION(:, :, :), POINTER :: ww1, ww2
   REAL(kind=realtype), DIMENSION(:, :, :), POINTER :: ww1b, ww2b
   REAL(kind=realtype), DIMENSION(:, :), POINTER :: pp1, pp2
   REAL(kind=realtype), DIMENSION(:, :), POINTER :: pp1b, pp2b
   REAL(kind=realtype), DIMENSION(:, :), POINTER :: rlv1, rlv2
   REAL(kind=realtype), DIMENSION(:, :), POINTER :: rlv1b, rlv2b
   REAL(kind=realtype), DIMENSION(:, :), POINTER :: rev1, rev2
   !
   !      Local variables
   !
   INTEGER(kind=inttype) :: id, ih
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
   ww1b => wb(ih, 1:, 1:, :)
   ww2b => wb(id, 1:, 1:, :)
   pp1b => pb(ih, 1:, 1:)
   pp2b => pb(id, 1:, 1:)
   IF (viscous) THEN
   rlv1b => rlvb(ih, 1:, 1:)
   rlv2b => rlvb(id, 1:, 1:)
   END IF
   CASE (imax) 
   !===============================================================
   ww1b => wb(ih, 1:, 1:, :)
   ww2b => wb(id, 1:, 1:, :)
   pp1b => pb(ih, 1:, 1:)
   pp2b => pb(id, 1:, 1:)
   IF (viscous) THEN
   rlv1b => rlvb(ih, 1:, 1:)
   rlv2b => rlvb(id, 1:, 1:)
   END IF
   CASE (jmin) 
   !===============================================================
   ww1b => wb(1:, ih, 1:, :)
   ww2b => wb(1:, id, 1:, :)
   pp1b => pb(1:, ih, 1:)
   pp2b => pb(1:, id, 1:)
   IF (viscous) THEN
   rlv1b => rlvb(1:, ih, 1:)
   rlv2b => rlvb(1:, id, 1:)
   END IF
   CASE (jmax) 
   !===============================================================
   ww1b => wb(1:, ih, 1:, :)
   ww2b => wb(1:, id, 1:, :)
   pp1b => pb(1:, ih, 1:)
   pp2b => pb(1:, id, 1:)
   IF (viscous) THEN
   rlv1b => rlvb(1:, ih, 1:)
   rlv2b => rlvb(1:, id, 1:)
   END IF
   CASE (kmin) 
   !===============================================================
   ww1b => wb(1:, 1:, ih, :)
   ww2b => wb(1:, 1:, id, :)
   pp1b => pb(1:, 1:, ih)
   pp2b => pb(1:, 1:, id)
   IF (viscous) THEN
   rlv1b => rlvb(1:, 1:, ih)
   rlv2b => rlvb(1:, 1:, id)
   END IF
   CASE (kmax) 
   !===============================================================
   ww1b => wb(1:, 1:, ih, :)
   ww2b => wb(1:, 1:, id, :)
   pp1b => pb(1:, 1:, ih)
   pp2b => pb(1:, 1:, id)
   IF (viscous) THEN
   rlv1b => rlvb(1:, 1:, ih)
   rlv2b => rlvb(1:, 1:, id)
   END IF
   END SELECT
   END SUBROUTINE SETBCPOINTERS_B