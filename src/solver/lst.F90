module lst
    use constants
#include <petsc/finclude/petsc.h>
    use petsc
#include <slepc/finclude/slepceps.h>
    use slepceps

    ! Module variables
    Mat drdw
    EPS epsContext

    ! Eigenvector stored for writing to file
    !real(kind=realType), dimension(:), allocatable :: lstEvecReal



contains
    subroutine initalizeLST
        ! Initialize SLEPc and create the matrices needed for the analysis
        use utils, only: EChk
        use inputTimeSpectral, only: nTimeIntervalsSpectral
        use flowVarRefState, only: nwf, nw, viscous
        use stencils, only: N_visc_drdw, visc_drdw_stencil, N_euler_drdw, euler_drdw_stencil
        use ADjointVars, only: nCellsLocal
        use inputADjoint, only: frozenTurbulence
        use adjointUtils, only: myMatCreate, statePreAllocation
        implicit none

        !
        !      Local variables
        !
        integer(kind=intType) :: ierr, level
        integer(kind=intType) :: nDimW
        integer(kind=intType) :: n_stencil, nState
        integer(kind=intType), dimension(:), allocatable :: nnzDiagonal, nnzOffDiag
        integer(kind=intType), dimension(:, :), pointer :: stencil

        ! Call destroy just in case we already initialized
        !call destroyLST()

        ! TODO: Do the inialization here for now, but should move probably to init step earlier
        call SlepcInitialize(PETSC_NULL_CHARACTER, "residual jacobian"//C_NEW_LINE, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        ! Evaluate sizes
        if (frozenTurbulence) then
            nState = nwf
        else
            nState = nw
        end if

        nDimW = nState * nCellsLocal(1_intType) * nTimeIntervalsSpectral

        ! Setup matrix-based dRdwT
        ! TODO: probably want to do this matrix-free (shell)
        allocate (nnzDiagonal(nCellsLocal(1_intType) * nTimeIntervalsSpectral), &
                    nnzOffDiag(nCellsLocal(1_intType) * nTimeIntervalsSpectral))

        if (viscous) then
            n_stencil = N_visc_drdw
            stencil => visc_drdw_stencil
        else
            n_stencil = N_euler_drdw
            stencil => euler_drdw_stencil
        end if

        level = 1_intType
        print *, "nDimW", nDimW
        call statePreAllocation(nnzDiagonal, nnzOffDiag, nDimW / nState, stencil, n_stencil, &
                                level, .False.)
        call myMatCreate(drdw, nState, nDimW, nDimW, nnzDiagonal, nnzOffDiag, &
                            __FILE__, __LINE__)

        call matSetOption(drdw, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        deallocate (nnzDiagonal, nnzOffDiag)

    end subroutine initalizeLST

    subroutine solveLST

        use inputADjoint, only: frozenTurbulence
        use adjointUtils, only: setupStateResidualMatrix
        use communication, only: ADFLOW_COMM_WORLD, myid
        use utils, only: EChk
        implicit none

        !
        !      Local variables
        !
        logical :: useAD, usePC, useTranspose, useObjective
        integer(kind=intType) :: ierr, level

        ! EPS variables
        PetscViewer viewer
        EPSType epsTypeName
        PetscInt its, nev, ncv, maxit
        PetscReal tol
        PetscLogDouble t1, t2

        ! Eigenvalue and eigenvector
        PetscScalar kr, ki
        Vec xr, xi


        ! Assembling the Jacobian dRdw
        ! Assemble with AD use the exact matrix, not the PC, and no transpose
        useAD = .True.
        usePC = .False.
        useTranspose = .False.
        useObjective = .False.
        level = 1_intType

        call setupStateResidualMatrix(drdw, useAD, usePC, useTranspose, &
                                      useObjective, frozenTurbulence, level)

        ! Visualize structure of matrix
        !call MatView(drdw, PETSC_VIEWER_DRAW_WORLD, ierr)
        !call PetscSleep(.0, ierr)

        ! Write the matrix to file for debugging
        call PetscViewerBinaryOpen(ADFLOW_COMM_WORLD, "drdw.petsc", FILE_MODE_WRITE, viewer, ierr)
        call EChk(ierr, __FILE__, __LINE__)
        call MatView(drdw, viewer, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        !call PetscViewerASCIIOpen(ADFLOW_COMM_WORLD,"drdw.ascii", viewer, ierr)
        !call PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB, ierr)
        !call EChk(ierr, __FILE__, __LINE__)
        !call MatView(drdw, viewer, ierr)
        !call EChk(ierr, __FILE__, __LINE__)

        call PetscViewerDestroy(viewer, ierr)
        call EChk(ierr, __FILE__, __LINE__)


        ! Create the eigensolver and set various options
        call EPSCreate(ADFLOW_COMM_WORLD, epsContext, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        ! Set operators. In this case, it is a standard eigenvalue problem
        call EPSSetOperators(epsContext, drdw, PETSC_NULL_MAT, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        ! Set the problem type. Default to non-hermitian EVP. Can overwrite with input param if know its e.g. symmetric
        call EPSSetProblemType(epsContext, EPS_NHEP, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        ! Set solution type
        call EPSSetWhichEigenpairs(epsContext, EPS_LARGEST_REAL, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        ! Set solution dimension
        nev = 1 ! Number of eigenvalues to compute. We only seek one eigenvalue
        ncv = 100 ! The maximum dimension of the subspace to be used by the solver
        call EPSSetDimensions(epsContext, nev, ncv, PETSC_DECIDE, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        ! Solve the eigensystem
        call PetscTime(t1, ierr)
        call EChk(ierr, __FILE__, __LINE__)
        call EPSSolve(epsContext, ierr)
        call EChk(ierr, __FILE__, __LINE__)
        call PetscTime(t2, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        call EPSView(epsContext, PETSC_VIEWER_STDOUT_WORLD, ierr)

        ! Extract solution information and print key information
        call EPSGetIterationNumber(epsContext, its, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        call EPSGetType(epsContext, epsTypeName, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        call EPSGetDimensions(epsContext, nev, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        call EPSGetTolerances(epsContext, tol, maxit, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        if (myid .eq. 0) then
            write (*, '(1X, A, 1X, f10.3)') "Eigenvalue solve time:", t2 - t1
            write (*, '(1X, A, 1X, I4)') "Number of iterations of the method:", its
            write (*, '(1X, A, 1X, A)') "Solution method:", epsTypeName
            write (*, '(1X, A, 1X, I4)') "Number of requested eigenvalues:", nev
            write (*, '(1X, A,E15.7,A,I6)') "Stopping condition: tol=", tol, " maxit=", maxit
        end if
        call PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL, ierr)
        call EPSConvergedReasonView(epsContext, PETSC_VIEWER_STDOUT_WORLD, ierr)
        call EPSErrorView(epsContext, EPS_ERROR_RELATIVE, PETSC_VIEWER_STDOUT_WORLD, ierr)
        call PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        ! Extract and store the first eigenvalue and eigenvector
        call MatCreateVecs(drdw, PETSC_NULL_VEC, xr, ierr)
        call EChk(ierr, __FILE__, __LINE__)
        call MatCreateVecs(drdw, PETSC_NULL_VEC, xi, ierr)
        call EChk(ierr, __FILE__, __LINE__)
        call EPSGetEigenpair(epsContext, 0, kr, ki, xr, xi, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        if (myid .eq. 0) then
            print *, "Eigenvalue real: ", kr
            print *, "Eigenvalue imag: ", ki
        end if

        ! Copy the eigenvector to the ADflow data structure
        call setEigenVector(xr)


    end subroutine solveLST

    subroutine destroyLST
        ! Deallocate any arrays that have been created in initalizeLST
        use utils, only: EChk
        implicit none

        ! Local variables
        integer(kind=intType) :: ierr

        call EPSDestroy(epsContext, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        call MatDestroy(drdw, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        call SlepcFinalize(ierr)
        call EChk(ierr, __FILE__, __LINE__)

    end subroutine destroyLST


    subroutine setEigenVector(eVec)
        ! This routine copies the eigenvector from the PETSc Vec to the ADflow data structure
        use constants
        use blockPointers, only: nDom, il, jl, kl, LSTEvecReal
        use inputTimeSpectral, only: nTimeIntervalsSpectral
        use flowVarRefState, only: nwf, nt1, nt2, winf
        use utils, only: setPointers, EChk

        implicit none

        Vec eVec
        integer(kind=intType) :: ierr, nn, sps, i, j, k, l, ii
        real(kind=realType), dimension(:), pointer :: eVecPointer

        call VecGetArrayReadF90(eVec, eVecPointer, ierr)
        call EChk(ierr, __FILE__, __LINE__)

        print * , "Max value of eigenvector: ", maxval(abs(eVecPointer))
        print * , "Min value of eigenvector: ", minval(abs(eVecPointer))

        ii = 1
        do nn = 1, nDom
            do sps = 1, nTimeIntervalsSpectral
                call setPointers(nn, 1_intType, sps)

                do k = 2, kl
                    do j = 2, jl
                        do i = 2, il
                            LSTEvecReal(i, j, k) = eVecPointer(ii)
                            print *, "Eigenvector i, val: ", ii, eVecPointer(ii)
                            ii = ii + 1
                        end do
                    end do
                end do
            end do
        end do

        call VecRestoreArrayReadF90(eVec, eVecPointer, ierr)
        call EChk(ierr, __FILE__, __LINE__)

    end subroutine setEigenVector

end module lst
