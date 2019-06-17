!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.10 (r5363) -  9 Sep 2014 09:53
!
SUBROUTINE GETFORCES_CD(forces, npts, sps)
  USE CONSTANTS_D
  USE COMMUNICATION_D, ONLY : myid
  USE BLOCKPOINTERS_D, ONLY : bcdata, ndom, nbocos, bctype
  USE INPUTPHYSICS_D, ONLY : forcesastractions
  USE UTILS_D, ONLY : setpointers, terminate, echk
  USE SURFACEINTEGRATIONS_D, ONLY : integratesurfaces
  USE SURFACEFAMILIES_D, ONLY : fullfamlist
  USE OVERSETDATA_D, ONLY : zippermeshes, zippermesh, oversetpresent
  USE SURFACEFAMILIES_D, ONLY : familyexchange, bcfamexchange
  USE PETSC_D
  IMPLICIT NONE
  INTEGER(kind=inttype), INTENT(IN) :: npts, sps
  REAL(kind=realtype), INTENT(INOUT) :: forces(3, npts)
  INTEGER(kind=inttype) :: mm, nn, i, j, ii, jj, idim, ierr
  INTEGER(kind=inttype) :: ibeg, iend, jbeg, jend
  REAL(kind=realtype) :: sss(3), v2(3), v1(3), qa, sepsensor, cavitation
  REAL(kind=realtype) :: sepsensoravg(3)
  REAL(kind=realtype) :: fp(3), fv(3), mp(3), mv(3), yplusmax, qf(3)
  TYPE(UNKNOWNTYPE) :: nlocalvalues
  REAL(kind=realtype) :: localvalues(nlocalvalues)
  TYPE(ZIPPERMESH), POINTER :: zipper
  TYPE(FAMILYEXCHANGE), POINTER :: exch
  REAL(kind=realtype), DIMENSION(:), POINTER :: localptr
  TYPE(UNKNOWNTYPE) :: ncostfunction
  REAL(kind=realtype), DIMENSION(ncostfunction) :: funcvalues
  EXTERNAL SETPOINTERS
  EXTERNAL INTEGRATESURFACES
  EXTERNAL COMPUTENODALTRACTIONS
  EXTERNAL COMPUTENODALFORCES
  EXTERNAL BCTYPE
  TYPE(UNKNOWNTYPE) :: BCTYPE
  EXTERNAL BCDATA
  TYPE UNKNOWNDERIVEDTYPE
      INTEGER :: jnbeg
      INTEGER :: jnend
      INTEGER :: inbeg
      INTEGER :: inend
      REAL, DIMENSION(:, :, :) :: tp
      REAL, DIMENSION(:, :, :) :: tv
      REAL, DIMENSION(:, :, :) :: f
      TYPE(UNKNOWNTYPE), DIMENSION(:) :: cellval
      TYPE(UNKNOWNTYPE), DIMENSION(:, :) :: area
      TYPE(UNKNOWNTYPE), DIMENSION(:) :: nodeval
      REAL, DIMENSION(:, :) :: cellheatflux
      REAL, DIMENSION(:, :) :: nodeheatflux
      REAL, DIMENSION(:, :, :) :: norm
  END TYPE UNKNOWNDERIVEDTYPE
  TYPE(UNKNOWNDERIVEDTYPE) :: BCDATA
  EXTERNAL ZIPPERMESHES
  TYPE(ZIPPERMESH) :: ZIPPERMESHES
  EXTERNAL BCFAMEXCHANGE
  TYPE(FAMILYEXCHANGE) :: BCFAMEXCHANGE
  EXTERNAL TERMINATE
  EXTERNAL VECGETARRAYF90
  EXTERNAL ECHK
  INTRINSIC SIZE
  EXTERNAL VECRESTOREARRAYF90
  EXTERNAL VECSCATTERBEGIN
  EXTERNAL VECSCATTEREND
  TYPE(UNKNOWNTYPE) :: result1
  TYPE(UNKNOWNTYPE) :: result2
  TYPE(UNKNOWNTYPE) :: result3
  TYPE(UNKNOWNDERIVEDTYPE) :: result10
  TYPE(UNKNOWNDERIVEDTYPE) :: result20
  TYPE(UNKNOWNTYPE) :: __file__
  TYPE(UNKNOWNTYPE) :: nswallisothermal
  TYPE(UNKNOWNTYPE) :: __line__
  TYPE(UNKNOWNTYPE) :: fullfamlist
  INTEGER :: nbocos
  INTEGER :: ndom
  TYPE(UNKNOWNTYPE) :: insert_values
  INTEGER :: myid
  TYPE(UNKNOWNTYPE) :: eulerwall
  TYPE(UNKNOWNTYPE) :: nswalladiabatic
  TYPE(UNKNOWNTYPE) :: ibcgroupwalls
  LOGICAL :: forcesastractions
  TYPE(UNKNOWNTYPE) :: scatter_forward
  REAL :: zero
  LOGICAL :: oversetpresent
! Make sure *all* forces are computed. Sectioning will be done
! else-where.
domains:DO nn=1,ndom
    CALL SETPOINTERS(nn, 1_intType, sps)
    localvalues = zero
    CALL INTEGRATESURFACES(localvalues, fullfamlist)
  END DO domains
  IF (forcesastractions) THEN
! Compute tractions if necessary
    CALL COMPUTENODALTRACTIONS(sps)
  ELSE
    CALL COMPUTENODALFORCES(sps)
  END IF
  ii = 0
domains2:DO nn=1,ndom
    CALL SETPOINTERS(nn, 1_intType, sps)
! Loop over the number of boundary subfaces of this block.
bocos:DO mm=1,nbocos
      result1 = BCTYPE(mm)
      result2 = BCTYPE(mm)
      result3 = BCTYPE(mm)
      IF ((result1 .EQ. eulerwall .OR. result2 .EQ. nswalladiabatic) &
&         .OR. result3 .EQ. nswallisothermal) THEN
! This is easy, just copy out F or T in continuous ordering.
        result10 = BCDATA(mm)
        result20 = BCDATA(mm)
        DO j=result10%jnbeg,result20%jnend
 100      result10 = BCDATA(mm)
          result20 = BCDATA(mm)
          i = result20%inend + 1
          result10 = BCDATA(mm)
          result20 = BCDATA(mm)
        END DO
      END IF
    END DO bocos
  END DO domains2
! We know must consider additional forces that are required by the
! zipper mesh triangles on the root proc.
! Pointer for easier reading.
  zipper => ZIPPERMESHES(ibcgroupwalls)
  exch => BCFAMEXCHANGE(ibcgroupwalls, sps)
! No overset present or the zipper isn't allocated nothing to do:
  IF ((.NOT.oversetpresent) .OR. (.NOT.zipper%allocated)) THEN
    RETURN
  ELSE
    IF (.NOT.forcesastractions) CALL TERMINATE('getForces', &
&                 'getForces() is not implmented for zipper meshes and '&
&                                        , 'forcesAsTractions=False')
! We have a zipper and regular forces are requested. This is not yet supported.
! Loop over each dimension individually since we have a scalar
! scatter.
dimloop:DO idim=1,3
      CALL VECGETARRAYF90(exch%nodevallocal, localptr, ierr)
      CALL ECHK(ierr, __file__, __line__)
! Copy in the values we already have to the exchange.
      ii = SIZE(localptr)
      localptr = forces(idim, 1:ii)
! Restore the pointer
      CALL VECRESTOREARRAYF90(exch%nodevallocal, localptr, ierr)
      CALL ECHK(ierr, __file__, __line__)
! Now scatter this to the zipper
      CALL VECSCATTERBEGIN(zipper%scatter, exch%nodevallocal, zipper%&
&                    localval, insert_values, scatter_forward, ierr)
      CALL ECHK(ierr, __file__, __line__)
      CALL VECSCATTEREND(zipper%scatter, exch%nodevallocal, zipper%&
&                  localval, insert_values, scatter_forward, ierr)
      CALL ECHK(ierr, __file__, __line__)
! The values we need are precisely what is in zipper%localVal
      CALL VECGETARRAYF90(zipper%localval, localptr, ierr)
      CALL ECHK(ierr, __file__, __line__)
! Just copy the received data into the forces array. Just on root proc:
      IF (myid .EQ. 0) forces(idim, ii+1:ii+SIZE(localptr)) = localptr
      CALL VECGETARRAYF90(zipper%localval, localptr, ierr)
      CALL ECHK(ierr, __file__, __line__)
    END DO dimloop
  END IF
END SUBROUTINE GETFORCES_CD
