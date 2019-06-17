!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.10 (r5363) -  9 Sep 2014 09:53
!
SUBROUTINE HEATFLUXES_CD()
  USE CONSTANTS_D
  USE BLOCKPOINTERS_D, ONLY : bcdata, ndom, nbocos, bctype, bcfaceid, &
& viscsubface
  USE BCPOINTERS_D, ONLY : ssi
  USE FLOWVARREFSTATE_D, ONLY : pref, rhoref
  USE UTILS_D, ONLY : setpointers, setbcpointers
  IMPLICIT NONE
!
!      Local variables.
!
  INTEGER(kind=inttype) :: i, j, ii, mm
  REAL(kind=realtype) :: fact, scaledim, q
  REAL(kind=realtype) :: qw, qa
  LOGICAL :: heatedsubface
  INTRINSIC SQRT
  EXTERNAL BCTYPE
  TYPE(UNKNOWNTYPE) :: BCTYPE
  EXTERNAL SETBCPOINTERS
  EXTERNAL BCFACEID
  TYPE(UNKNOWNTYPE) :: BCFACEID
  EXTERNAL BCDATA
  TYPE UNKNOWNDERIVEDTYPE
      REAL, DIMENSION(:, :, :) :: q
  END TYPE UNKNOWNDERIVEDTYPE
  TYPE(UNKNOWNDERIVEDTYPE) :: BCDATA
  EXTERNAL SSI
  REAL :: SSI
  EXTERNAL VISCSUBFACE
  TYPE(UNKNOWNDERIVEDTYPE) :: VISCSUBFACE
  REAL :: arg1
  REAL :: result1
  TYPE(UNKNOWNTYPE) :: result10
  TYPE(UNKNOWNTYPE) :: result11
  TYPE(UNKNOWNDERIVEDTYPE) :: result12
  TYPE(UNKNOWNDERIVEDTYPE) :: result2
  REAL :: result20
  REAL :: result3
  REAL :: result4
  REAL :: rhoref
  TYPE(UNKNOWNTYPE) :: nswallisothermal
  REAL :: one
  REAL :: pref
  TYPE(UNKNOWNTYPE) :: imin
  INTEGER :: nbocos
  TYPE(UNKNOWNTYPE) :: jmin
  TYPE(UNKNOWNTYPE) :: kmin
  TYPE(UNKNOWNTYPE) :: imax
  TYPE(UNKNOWNTYPE) :: jmax
  TYPE(UNKNOWNTYPE) :: kmax
! Set the actual scaling factor such that ACTUAL heat flux is computed
! The factor is determined from stanton number
  arg1 = pref/rhoref
  result1 = SQRT(arg1)
  scaledim = pref*result1
! Loop over the boundary subfaces of this block.
bocos:DO mm=1,nbocos
! Only do this on isoThermalWalls
    result10 = BCTYPE(mm)
    IF (result10 .EQ. nswallisothermal) THEN
! Set a bunch of pointers depending on the face id to make
! a generic treatment possible. The routine setBcPointers
! is not used, because quite a few other ones are needed.
      CALL SETBCPOINTERS(mm, .true.)
      result11 = BCFACEID(mm)
      SELECT CASE  (result11) 
      CASE (imin, jmin, kmin) 
        fact = -one
      CASE (imax, jmax, kmax) 
        fact = one
      END SELECT
! Loop over the quadrilateral faces of the subface. Note that
! the nodal range of BCData must be used and not the cell
! range, because the latter may include the halo's in i and
! j-direction. The offset +1 is there, because inBeg and jnBeg
! refer to nodal ranges and not to cell ranges.
!
      result12 = BCDATA(mm)
      result2 = BCDATA(mm)
      DO j=result12%jnbeg+1,result2%jnend
 100    result12 = BCDATA(mm)
        result2 = BCDATA(mm)
        i = result2%inend + 1
        result12 = BCDATA(mm)
        result2 = BCDATA(mm)
      END DO
    END IF
  END DO bocos
END SUBROUTINE HEATFLUXES_CD