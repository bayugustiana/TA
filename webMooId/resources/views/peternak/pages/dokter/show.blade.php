@extends('peternak.layouts.app', ['activeMenu' => 'dokter'])

@section('hstyles')

@endsection

@section('content')
<div class="pcoded-content">
    <!-- PAGE HEADER -->
    <div class="page-header">
        <div class="page-block">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <div class="page-header-title">
                        <h5 class="m-b-10">Tampil Dokter</h5>
                        <p class="m-b-0">Tampilan Identitas Dokter</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Dokter</a></li>
                        <li class="breadcrumb-item"><a href="#">Tampil Dokter</a></li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    <div class="pcoded-inner-content">
        <div class="main-body">
            <div class="page-wrapper">

                <div class="page-body">

                    <div class="row">
                        <div class="col-lg-12">
                            <div class="cover-profile">
                                <div class="profile-bg-img">
                                    <img class="profile-bg-img img-fluid"
                                        src="{{ asset('assets/images/user-profile/bg-img1.jpg') }}" alt="bg-img">
                                    <div class="card-block user-info">
                                        <div class="col-md-12">
                                            <div class="media-left">
                                                <a href="#" class="profile-image">
                                                    <img class="user-img img-radius"
                                                        src="{{ (@$info->avatar) ? storage_path('public/dokter/' . $info->avatar) : asset('assets/images/avatar-0.png') }}" alt="user-img"
                                                        width="100px">
                                                </a>
                                            </div>
                                            <div class="media-body row">
                                                <div class="col-lg-12">
                                                    <div class="user-title">
                                                        <h2>{{ $info->nama }}</h2>
                                                        <span class="text-white">Dokter Hewan</span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="row">
                        <div class="col-lg-12">

                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-header-text">Tentang</h5>
                                    <form action="{{ route('dokter.destroy' , $info->id)}}" method="POST">
                                        @method('DELETE')
                                        {{ csrf_field() }}
                                        <button class="btn btn-sm btn-danger waves-effect waves-light f-right"><i class="fas fa-trash"></i></button>
                                        <a href="{{ route('dokter.edit', $info->id) }}" class="btn btn-sm btn-warning waves-effect waves-light f-right"><i class="fas fa-user-edit"></i></a>
                                    </form>
                                </div>
                                <div class="card-block">
                                    <div class="view-info">
                                        <div class="row">
                                            <div class="col-lg-12">
                                                <div class="general-info">
                                                    <div class="row">
                                                        <div class="col-lg-12 col-xl-6">
                                                            <div class="table-responsive">
                                                                <table class="table m-0">
                                                                    <tbody>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Nama Lengkap
                                                                            </th>
                                                                            <td>{{ $info->nama }}</td>
                                                                        </tr>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Email</th>
                                                                            <td>{{ $info->email }}</td>
                                                                        </tr>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Nomor Telepon
                                                                            </th>
                                                                            <td>{{ $info->no_telepon }}</td>
                                                                        </tr>
                                                                    </tbody>
                                                                </table>
                                                            </div>
                                                        </div>

                                                        <div class="col-lg-12 col-xl-6">
                                                            <div class="table-responsive">
                                                                <table class="table">
                                                                    <tbody>

                                                                        <tr>
                                                                            <th scope="row">
                                                                                Pendidikan Terakhir</th>
                                                                            <td>{{ $info->pendidikan_terakhir }}</td>
                                                                        </tr>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Pengalaman Kerja
                                                                            </th>
                                                                            <td>{{ $info->pengalaman_kerja }} Bulan</td>
                                                                        </tr>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Tempat Kerja
                                                                            </th>
                                                                            <td>{{ $info->tempat_kerja }}</td>
                                                                        </tr>
                                                                    </tbody>
                                                                </table>
                                                            </div>
                                                        </div>

                                                    </div>

                                                </div>

                                            </div>

                                        </div>

                                    </div>

                                </div>

                            </div>

                        </div>

                    </div>

                    <div class="row">
                        <div class="col-lg-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-header-text">Deskripsi Saya</h5>
                                </div>
                                <div class="card-block user-desc">
                                    <div class="view-desc">
                                        {!! $info->tentang !!}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </div>

    </div>
</div>
</div>
@endsection

@section('fscripts')

@endsection
