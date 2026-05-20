! . /etc/profile.d/modules.sh && module load nvhpc/25.1
! nvfortran -O2 -acc -gpu=cc80,cuda12.6 -mcmodel=medium -Mextend -c keme.f -o keme_oacc.o
! nvfortran -O2 -acc -gpu=cc80,cuda12.6 -mcmodel=medium -Mextend -c pointsource.f -o pointsource_oacc.o
! mpifort -O2 -acc -gpu=cc80,cuda12.6 -mcmodel=medium main_oacc.f90 keme_oacc.o pointsource_oacc.o /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3

program main
   use mpi
   use openacc
   implicit none

   ! ----- 問題パラメータ -----
   ! integer, parameter :: nex = 101, ney = 101, nez = 50
   ! integer, parameter :: nex = 201, ney = 201, nez = 100
   ! integer, parameter :: nex = 301, ney = 301, nez = 150
   integer, parameter :: nex = 721, ney = 721, nez = 350
   integer, parameter :: kd = 2, nt = 100000, nobs = 3
   integer, parameter :: layer_size = (nex+1)*(ney+1)
   character(len=*), parameter :: output_file = './4.dat'

   ! ----- MPI / 分割 -----
   integer :: nproc, myrank, ierr
   integer :: k_eL, k_eR, nez_local, nz_local, nz_owned
   integer :: n_local, ne_local, ne_interior
   integer :: neighbor_below, neighbor_above
   integer :: src_rank, ie_src_local
   integer :: req_un(2), req_up(2)
   integer :: statuses(MPI_STATUS_SIZE, 2)

   ! ----- 物性・震源 -----
   real*8 :: c1(kd), c2(kd), rho(kd), Kl(kd), Gl(kd)
   real*8 :: ds, dt, rt, strike, dip, rake, moment, tim, pi
   real*8 :: fai, delta, ramda, coem(3,3), fault(3)
   real*8 :: source(nt), ft_const(96), kek(96,96), keg(96,96)
   real*8 :: rmtmp_all(4,4,8,kd)
   integer :: ifx, ify, ifz

   ! ----- 観測点 -----
   real*8 :: obs(2,nobs), tobs(3,nobs)
   real*8 :: obs_local(3,nobs), obs_buf(3,nobs)
   integer :: iobs_local(nobs)

   ! ----- ローカル動的配列 -----
   real*8, allocatable :: rm(:,:,:)
   real*8, allocatable :: un(:), um(:), up(:)
   integer, allocatable :: cny(:,:), num(:)
   integer, allocatable :: flag(:,:,:)
   real*8, allocatable :: recv_up_buf(:)
   real*8, allocatable :: recv_rm_buf(:,:,:)

   ! ----- 作業用 -----
   real*8 :: rmt(4,4), uns(4,4), vns(4,4), sns(4), snsm(4,4), rmtinv(4,4)
   real*8 :: upm(4,3), fpm(4,3), val
   integer :: i, j, k, ii, jj, kk, i1, i2, j2, id, ie, in, it
   integer :: k_global, k_e_local, r, kL_tmp, kR_tmp, k_e_local_top

   call MPI_Init(ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nproc, ierr)
   call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)

   ! ====== 各 MPI ランクを自分の GPU に割り当て ======
   block
      integer :: local_rank_, ndev_, mygpu_
      character(len=32) :: lr_str_
      call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK', lr_str_)
      if (len_trim(lr_str_) > 0) then
         read(lr_str_, *) local_rank_
      else
         local_rank_ = myrank
      endif
      ndev_ = acc_get_num_devices(acc_device_nvidia)
      if (ndev_ > 0) then
         mygpu_ = mod(local_rank_, ndev_)
         call acc_set_device_num(mygpu_, acc_device_nvidia)
         write(6,'(a,i0,a,i0,a,i0,a,i0)') &
            'rank ', myrank, ': local_rank=', local_rank_, &
            ', ndev=', ndev_, ', gpu=', mygpu_
      endif
   end block

   ! ====== z-スラブ分割 ======
   call partition_z_inline(nez, nproc, myrank, k_eL, k_eR)
   nez_local = k_eR - k_eL + 1
   nz_local  = nez_local + 1                    ! 上端1層を予約 (r<nproc-1 ではゴースト)
   if (myrank == nproc - 1) then
      nz_owned    = nz_local                    ! 最後ランクは top 層も所有
      ne_interior = nez_local * nex * ney       ! ゴースト依存要素なし
   else
      nz_owned    = nez_local                   ! 上端 nz_local は r+1 が所有 (ゴースト)
      ne_interior = max(0, (nez_local - 1) * nex * ney)
   endif
   n_local  = layer_size * nz_local
   ne_local = nex * ney * nez_local

   neighbor_below = MPI_PROC_NULL
   neighbor_above = MPI_PROC_NULL
   if (myrank > 0)         neighbor_below = myrank - 1
   if (myrank < nproc - 1) neighbor_above = myrank + 1

   if (myrank == 0) then
      write(6,'(a,i0,a,i0,a,i0,a,i0)') &
         'mesh: nex=', nex, ', ney=', ney, ', nez=', nez, ', nproc=', nproc
   endif
   write(6,'(a,i0,a,i0,a,i0,a,i0,a,i0)') &
      'rank ', myrank, ': k_eL=', k_eL, ', k_eR=', k_eR, &
      ', nez_local=', nez_local, ', ne_interior=', ne_interior

   ! ====== 配列確保 ======
   allocate(rm(4,4,n_local))
   allocate(un(12*n_local), um(12*n_local), up(12*n_local))
   allocate(cny(8,ne_local), num(ne_local))
   allocate(flag(nex+1, ney+1, nz_local))
   allocate(recv_up_buf(12*layer_size))
   allocate(recv_rm_buf(4,4,layer_size))

   ! ====== 物性 ======
   ds = 270.d0
   dt = 0.012d0
   c1(1) = 3900.d0; c2(1) = 2250.d0; rho(1) = 2500.d0
   c1(2) = 7800.d0; c2(2) = 4500.d0; rho(2) = 3000.d0

   ! 現状は要素中心に設定する
   fault(1) = (nex+1)/2*ds - 0.5d0*ds
   fault(2) = (ney+1)/2*ds - 0.5d0*ds
   fault(3) = nez*ds - 2025.d0

   do i = 1, nobs
      obs(1,i) = fault(1) + 4725.d0 + 2700.d0*(i-1)
      obs(2,i) = fault(2) + 4725.d0 + 2700.d0*(i-1)
   enddo

   ! risetime
   rt = 2.0d0
   strike = 30.d0; dip = 40.d0; rake = 50.d0; moment = 1.0d15

   ifx = int(fault(1)/ds) + 1
   ify = int(fault(2)/ds) + 1
   ifz = int(fault(3)/ds) + 1

   source = 1.d0
   do it = 1, nt
      tim = dt * (it-1)
      if (tim .le. rt/2.d0) source(it) = 2.d0*tim*tim/rt/rt
      if ((tim .gt. rt/2.d0) .and. (tim .le. rt)) source(it) = 1.d0 - 2.d0*(tim-rt)**2/rt/rt
   enddo
   source = source * moment

   pi = 4.d0 * atan(1.d0)
   fai = pi/180.d0 * strike
   delta = pi/180.d0 * dip
   ramda = pi/180.d0 * rake
   call coemoment(delta, ramda, fai, coem)

   do i = 1, kd
      Gl(i) = rho(i) * c2(i)**2
      Kl(i) = rho(i) * c1(i)**2 - 4.d0/3.d0 * Gl(i)
   enddo

   ! ====== 材料ごとに要素質量行列を 1 回ずつ計算 (一様メッシュ・均質材料を仮定) ======
   do k = 1, kd
      call cmp_me(ds, rho(k), rmtmp_all(:,:,:,k))
   enddo

   ! ====== 震源ランクの特定と局所要素インデックス ======
   src_rank = -1
   do r = 0, nproc - 1
      call partition_z_inline(nez, nproc, r, kL_tmp, kR_tmp)
      if (ifz >= kL_tmp .and. ifz <= kR_tmp) then
         src_rank = r
         exit
      endif
   enddo
   if (myrank == src_rank) then
      k_e_local = ifz - k_eL + 1
      ie_src_local = ifx + (ify-1)*nex + (k_e_local-1)*nex*ney
   else
      ie_src_local = -1
   endif

   ! ====== 観測点 (自由表面 z=zmax は最後ランクが所有, flag の閉形式から直接計算) ======
   if (myrank == 0) write(6,*) 'obs point (x,y,z)'
   k_e_local_top = (nez+1) - k_eL + 1
   do i = 1, nobs
      i1 = int(obs(1,i)/ds) + 1
      i2 = int(obs(2,i)/ds) + 1
      tobs(1,i) = (i1-1)*ds
      tobs(2,i) = (i2-1)*ds
      tobs(3,i) = nez*ds
      if (myrank == nproc - 1) then
         iobs_local(i) = i1 + (i2-1)*(nex+1) + (k_e_local_top-1)*layer_size
      else
         iobs_local(i) = -1
      endif
      if (myrank == 0) write(6,*) tobs(1:3,i)
   enddo
   if (myrank == 0) then
      write(6,*) 'fault location (x,y,z)'
      write(6,*) (ifx-1)*ds + 0.5d0*ds, (ify-1)*ds + 0.5d0*ds, (ifz-1)*ds + 0.5d0*ds
      write(6,*) 'relative (x,y,z)'
      do i = 1, nobs
         write(6,*) tobs(1,i) - ((ifx-1)*ds + 0.5d0*ds), &
            tobs(2,i) - ((ify-1)*ds + 0.5d0*ds), &
            tobs(3,i) - ((ifz-1)*ds + 0.5d0*ds)
      enddo
   endif

   call def_ke(ds, kek, keg)
   call cmp_eff_fault(coem, ds, ft_const)
   ft_const = -ft_const

   un = 0.d0
   um = 0.d0

   if (myrank == 0) then
      open(60, file=output_file, status='unknown')
   endif

   ! ====== OpenACC データ領域 ======
   !$acc data copyin(kek, keg, Kl, Gl, source, ft_const, rmtmp_all, iobs_local, un, um) create(flag, cny, num, rm, up, recv_up_buf, recv_rm_buf, obs_local)

   ! ====== [GPU 初期化 1] flag(i,j,k) = i + (j-1)*(nex+1) + (k-1)*layer_size ======
   !$acc parallel loop collapse(3)
   do k = 1, nz_local
      do j = 1, ney+1
         do i = 1, nex+1
            flag(i,j,k) = i + (j-1)*(nex+1) + (k-1)*layer_size
         enddo
      enddo
   enddo

   ! ====== [GPU 初期化 2] cny / num を並列構築 ======
   !$acc parallel loop collapse(3) private(ie, k_global)
   do k_e_local = 1, nez_local
      do j = 1, ney
         do i = 1, nex
            ie = i + (j-1)*nex + (k_e_local-1)*nex*ney
            k_global = k_eL + k_e_local - 1
            if (((k_global-1)*ds + 0.5d0*ds) .gt. nez*ds - 2700.d0) then
               num(ie) = 1
            else
               num(ie) = 2
            endif
            cny(1,ie) = flag(i,   j,   k_e_local)
            cny(2,ie) = flag(i+1, j,   k_e_local)
            cny(3,ie) = flag(i+1, j+1, k_e_local)
            cny(4,ie) = flag(i,   j+1, k_e_local)
            cny(5,ie) = flag(i,   j,   k_e_local+1)
            cny(6,ie) = flag(i+1, j,   k_e_local+1)
            cny(7,ie) = flag(i+1, j+1, k_e_local+1)
            cny(8,ie) = flag(i,   j+1, k_e_local+1)
         enddo
      enddo
   enddo

   ! ====== [GPU 初期化 3] rm を 0 にゼロクリア ======
   !$acc parallel loop collapse(3)
   do id = 1, n_local
      do j = 1, 4
         do i = 1, 4
            rm(i,j,id) = 0.d0
         enddo
      enddo
   enddo

   ! ====== [GPU 初期化 4] 質量行列のアセンブル (atomic update) ======
   !$acc parallel loop private(in, i1, id)
   do ie = 1, ne_local
      in = num(ie)
      !$acc loop seq
      do i1 = 1, 8
         id = cny(i1, ie)
         !$acc loop seq
         do j = 1, 4
            !$acc loop seq
            do i = 1, 4
               !$acc atomic update
               rm(i,j,id) = rm(i,j,id) + rmtmp_all(i,j,i1,in)
            enddo
         enddo
      enddo
   enddo

   ! ====== [GPU 初期化 5] rm 共有層を GPU-direct でハロー交換 ======
   if (nproc > 1) then
      !$acc host_data use_device(rm, recv_rm_buf)
      call MPI_Sendrecv( &
         rm(1,1,(nz_local-1)*layer_size+1), 16*layer_size, MPI_DOUBLE_PRECISION, &
         neighbor_above, 200, &
         recv_rm_buf(1,1,1), 16*layer_size, MPI_DOUBLE_PRECISION, &
         neighbor_below, 200, &
         MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
      !$acc end host_data
      if (myrank > 0) then
         !$acc parallel loop collapse(3)
         do k = 1, layer_size
            do j = 1, 4
               do i = 1, 4
                  rm(i,j,k) = rm(i,j,k) + recv_rm_buf(i,j,k)
               enddo
            enddo
         enddo
      endif
   endif

   ! ====== [CPU SVD] LAPACK 制約のため host で 4x4 逆行列化、終了後 device に同期 ======
   !$acc update host(rm)

   snsm = 0.d0
   do id = 1, nz_owned * layer_size
      do j = 1, 4
         do i = 1, 4
            rmt(i,j) = rm(i,j,id)
         enddo
      enddo
      call svd_mgtn(4, 4, rmt, uns, vns, sns)
      if (abs(sns(4)/sns(1)) .le. 1.0d-6) then
         write(6,*) 'rm singular at rank=', myrank, ' local_id=', id
         call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      endif
      do i = 1, 4
         snsm(i,i) = 1.d0 / sns(i)
      enddo
      rmtinv = matmul(snsm, transpose(uns))
      rmtinv = matmul(transpose(vns), rmtinv)
      do j = 1, 4
         do i = 1, 4
            rm(i,j,id) = rmtinv(i,j)
         enddo
      enddo
   enddo

   !$acc update device(rm)

   ! ====== 時間ループ ======
   do it = 1, nt
      if (myrank == 0) write(6,*) it

      ! [1] un ゴースト交換を非同期で起動
      !     un は前ステップ末で更新済み → 境界要素 assembly までに必要
      if (nproc > 1) then
         !$acc host_data use_device(un)
         call MPI_Isend(un(1), 12*layer_size, MPI_DOUBLE_PRECISION, &
            neighbor_below, 10, MPI_COMM_WORLD, req_un(1), ierr)
         call MPI_Irecv(un(12*(nz_local-1)*layer_size + 1), 12*layer_size, &
            MPI_DOUBLE_PRECISION, neighbor_above, 10, &
            MPI_COMM_WORLD, req_un(2), ierr)
         !$acc end host_data
      else
         req_un(1) = MPI_REQUEST_NULL
         req_un(2) = MPI_REQUEST_NULL
      endif

      ! [2] up = 0
      !$acc parallel loop
      do i = 1, 12*n_local
         up(i) = 0.d0
      enddo

      ! [3] 内部要素アセンブリ (ゴースト un 不要・通信と並走)
      !$acc parallel loop private(val,in,i1,i2,ii,jj,j2,kk)
      do ie = 1, ne_interior
         in = num(ie)
         !$acc loop seq
         do i1 = 1, 8
            i2 = cny(i1, ie)
            !$acc loop seq
            do ii = 1, 12
               val = 0.d0
               !$acc loop seq
               do jj = 1, 8
                  j2 = cny(jj, ie)
                  !$acc loop seq
                  do kk = 1, 12
                     val = val + (Kl(in)*kek(12*(i1-1)+ii, 12*(jj-1)+kk) &
                        + Gl(in)*keg(12*(i1-1)+ii, 12*(jj-1)+kk)) &
                        * un(12*(j2-1)+kk)
                  enddo
               enddo
               if (ie == ie_src_local) then
                  val = val + ft_const(12*(i1-1)+ii) * source(it)
               endif
               !$acc atomic update
               up(12*(i2-1)+ii) = up(12*(i2-1)+ii) + val
            enddo
         enddo
      enddo

      ! [4] un 交換を完了
      if (nproc > 1) call MPI_Waitall(2, req_un, statuses, ierr)

      ! [5] 境界要素アセンブリ (top ゴースト un を読み、top ゴースト up に書く)
      if (ne_local > ne_interior) then
         !$acc parallel loop private(val,in,i1,i2,ii,jj,j2,kk)
         do ie = ne_interior + 1, ne_local
            in = num(ie)
            !$acc loop seq
            do i1 = 1, 8
               i2 = cny(i1, ie)
               !$acc loop seq
               do ii = 1, 12
                  val = 0.d0
                  !$acc loop seq
                  do jj = 1, 8
                     j2 = cny(jj, ie)
                     !$acc loop seq
                     do kk = 1, 12
                        val = val + (Kl(in)*kek(12*(i1-1)+ii, 12*(jj-1)+kk) &
                           + Gl(in)*keg(12*(i1-1)+ii, 12*(jj-1)+kk)) &
                           * un(12*(j2-1)+kk)
                     enddo
                  enddo
                  if (ie == ie_src_local) then
                     val = val + ft_const(12*(i1-1)+ii) * source(it)
                  endif
                  !$acc atomic update
                  up(12*(i2-1)+ii) = up(12*(i2-1)+ii) + val
               enddo
            enddo
         enddo
      endif

      ! [6] up ゴースト交換: 自分の top 層 (nz_local) を r+1 へ、r-1 から受信して layer 1 に加算
      if (nproc > 1) then
         !$acc host_data use_device(up, recv_up_buf)
         call MPI_Isend(up(12*(nz_local-1)*layer_size + 1), 12*layer_size, &
            MPI_DOUBLE_PRECISION, neighbor_above, 20, &
            MPI_COMM_WORLD, req_up(1), ierr)
         call MPI_Irecv(recv_up_buf(1), 12*layer_size, MPI_DOUBLE_PRECISION, &
            neighbor_below, 20, MPI_COMM_WORLD, req_up(2), ierr)
         !$acc end host_data
         call MPI_Waitall(2, req_up, statuses, ierr)
         if (myrank > 0) then
            !$acc parallel loop
            do i = 1, 12*layer_size
               up(i) = up(i) + recv_up_buf(i)
            enddo
         endif
      endif

      ! [7] 境界条件 — 底面 (rank 0 のみ)
      if (myrank == 0) then
         !$acc parallel loop collapse(2)
         do j = 1, ney+1
            do i = 1, nex+1
               id = flag(i, j, 1)
               !$acc loop seq
               do ii = 1, 12
                  up(12*(id-1)+ii) = 0.d0
               enddo
            enddo
         enddo
      endif

      ! 側面: y=0, y=ymax
      !$acc parallel loop collapse(2)
      do k = 1, nz_owned
         do i = 1, nex+1
            id = flag(i, 1, k)
            !$acc loop seq
            do ii = 1, 12
               up(12*(id-1)+ii) = 0.d0
            enddo
            id = flag(i, ney+1, k)
            !$acc loop seq
            do ii = 1, 12
               up(12*(id-1)+ii) = 0.d0
            enddo
         enddo
      enddo

      ! 側面: x=0, x=xmax
      !$acc parallel loop collapse(2)
      do k = 1, nz_owned
         do j = 1, ney+1
            id = flag(1, j, k)
            !$acc loop seq
            do ii = 1, 12
               up(12*(id-1)+ii) = 0.d0
            enddo
            id = flag(nex+1, j, k)
            !$acc loop seq
            do ii = 1, 12
               up(12*(id-1)+ii) = 0.d0
            enddo
         enddo
      enddo

      ! [8] 質量逆行列の適用 + 中心差分時間更新 (所有節点のみ)
      !$acc parallel loop private(upm,fpm,i,j,ii)
      do id = 1, nz_owned * layer_size
         !$acc loop seq
         do ii = 1, 12
            up(12*(id-1)+ii) = -up(12*(id-1)+ii) * dt * dt
         enddo
         !$acc loop seq
         do i = 1, 4
            !$acc loop seq
            do ii = 1, 3
               upm(i,ii) = up(12*(id-1) + 3*(i-1) + ii)
            enddo
         enddo
         !$acc loop seq
         do i = 1, 4
            !$acc loop seq
            do ii = 1, 3
               fpm(i,ii) = 0.d0
               !$acc loop seq
               do j = 1, 4
                  fpm(i,ii) = fpm(i,ii) + rm(i,j,id) * upm(j,ii)
               enddo
            enddo
         enddo
         !$acc loop seq
         do i = 1, 4
            !$acc loop seq
            do ii = 1, 3
               up(12*(id-1) + 3*(i-1) + ii) = fpm(i,ii)
            enddo
         enddo
         !$acc loop seq
         do ii = 1, 12
            up(12*(id-1)+ii) = up(12*(id-1)+ii) &
               + 2.d0 * un(12*(id-1)+ii) - um(12*(id-1)+ii)
            um(12*(id-1)+ii) = un(12*(id-1)+ii)
            un(12*(id-1)+ii) = up(12*(id-1)+ii)
         enddo
      enddo

      ! [9] 観測点抽出 (最後ランクのみ非ゼロ、他は 0)
      if (myrank == nproc - 1) then
         !$acc parallel loop collapse(2)
         do i = 1, nobs
            do ii = 1, 3
               obs_local(ii,i) = up(12*(iobs_local(i)-1) + ii)
            enddo
         enddo
         !$acc update host(obs_local)
      else
         obs_local = 0.d0
      endif

      ! [10] Allreduce で集約
      call MPI_Allreduce(obs_local, obs_buf, 3*nobs, MPI_DOUBLE_PRECISION, MPI_SUM, &
         MPI_COMM_WORLD, ierr)

      ! [11] rank 0 のみが output_file に書き込み
      if (myrank == 0) then
         do i = 1, nobs
            write(6,*)  obs_buf(1,i), obs_buf(2,i), obs_buf(3,i)
            write(60,*) obs_buf(1,i), obs_buf(2,i), obs_buf(3,i)
         enddo
      endif
   enddo

   !$acc end data

   if (myrank == 0) close(60)

   call MPI_Finalize(ierr)

contains

   subroutine partition_z_inline(nez_in, nproc_in, myrank_in, kL_out, kR_out)
      implicit none
      integer, intent(in) :: nez_in, nproc_in, myrank_in
      integer, intent(out) :: kL_out, kR_out
      integer :: base, rem
      base = nez_in / nproc_in
      rem  = mod(nez_in, nproc_in)
      if (myrank_in < rem) then
         kL_out = myrank_in * (base + 1) + 1
         kR_out = kL_out + base
      else
         kL_out = rem * (base + 1) + (myrank_in - rem) * base + 1
         kR_out = kL_out + base - 1
      endif
   end subroutine partition_z_inline

   subroutine coemoment(DIP, RAK, STR, coem)
      real*8 coem(3,3), DIP, RAK, STR
      coem(1,1) = 2*Cos(STR)*Sin(DIP) * &
         (-(Cos(DIP)*Cos(STR)*Sin(RAK)) + Cos(RAK)*Sin(STR))
      coem(2,2) = -2*Sin(DIP)*Sin(STR) * &
         (Cos(RAK)*Cos(STR) + Cos(DIP)*Sin(RAK)*Sin(STR))
      coem(3,3) = 2*Cos(DIP)*Sin(DIP)*Sin(RAK)
      coem(1,2) = Sin(DIP) * (Cos(RAK)*Cos(2*STR) + Cos(DIP)*Sin(RAK)*Sin(2*STR))
      coem(1,3) = -(Cos(DIP)**2*Cos(STR)*Sin(RAK)) + &
         Cos(STR)*Sin(DIP)**2*Sin(RAK) + Cos(DIP)*Cos(RAK)*Sin(STR)
      coem(2,3) = Cos(DIP)*Cos(RAK)*Cos(STR) + Cos(DIP)**2*Sin(RAK)*Sin(STR) - &
         Sin(DIP)**2*Sin(RAK)*Sin(STR)
      coem(2,1) = coem(1,2)
      coem(3,2) = coem(2,3)
      coem(3,1) = coem(1,3)
   end subroutine coemoment

   subroutine svd_mgtn(m, n, a, un, v, sn)
      integer m, n
      real*8 a(m,n)
      integer info
      character jobu, jobv
      integer lda, ldu, ldv, lwork
      real*8 sn(n), un(m,n), v(n,n)
      real*8 work(5*n + m)
      jobu = 's'
      jobv = 'a'
      lda  = m
      ldu  = m
      ldv  = n
      lwork = 5*n + m
      call dgesvd(jobu, jobv, m, n, a, lda, sn, un, ldu, v, ldv, work, lwork, info)
      if (info .gt. 0) then
         write(*,*) 'The algorithm computing SVD failed to converge.'
         stop
      endif
   end subroutine svd_mgtn

end program main
