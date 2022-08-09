@extends('peternak.layouts.app', ['activeMenu' => 'toko_obat'])

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
                        <h5 class="m-b-10">Toko Obat</h5>
                        <p class="m-b-0">Data Toko Obat</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Toko Obat</a>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    <!-- PAGE INNER CONTENT -->
    <div class="pcoded-inner-content">
        <div class="main-body">
            <div class="page-wrapper">
                <div class="page-body">
                    <!-- HERE IS THE EXAMPLE CONTENT-->
                    <div class="row">
                        <div class="col-lg-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Data Seluruh Toko Obat</h5>
                                    <div class="card-header-right">
                                        <ul class="list-unstyled card-option">
                                            <li><i class="fa fa fa-wrench open-card-option"></i>
                                            </li>
                                            <li><i class="fa fa-window-maximize full-card"></i></li>
                                            <li><i class="fa fa-minus minimize-card"></i></li>
                                            <li><i class="fa fa-refresh reload-card"></i></li>
                                            <li><i class="fa fa-trash close-card"></i></li>
                                        </ul>
                                    </div>
                                </div>
                                <div class="card-block">
                                    <div class="row mb-3">
                                        <div class="col-xl-12">
                                            <a href="{{ route('toko_obat.create') }}" class="btn btn-sm btn-primary"><i class="fas fa-plus"></i> Tambah</a>
                                        </div>
                                    </div>
                                    <div class="row">
                                        <div class="col-xl-12">
                                            <div class="dt-responsive table-responsive">
                                                <table id="simpletable"
                                                    class="table table-striped table-bordered nowrap">
                                                    <thead>
                                                        <tr>
                                                            <th>Image</th>
                                                            <th>Nama Toko Obat</th>
                                                            <th>Nama Pemilik</th>
                                                            <th>Email</th>
                                                            <th>No Telepon</th>
                                                            <th>Aksi</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        @if(@$info_toko_obat)
                                                        @foreach ($info_toko_obat as $info_data)
                                                        <tr>
                                                            <td><img class="user-img img-radius"
                                                                src="{{ (@$info_data->avatar) ? storage_path('public/toko_obat/' . $info_data->avatar) : asset('assets/images/avatar-0.png') }}" alt="user-img"
                                                                width="50"></td>
                                                            <td>{{ $info_data->nama_toko_obat }}</td>
                                                            <td>{{ $info_data->nama_lengkap }}</td>
                                                            <td>{{ $info_data->email }}</td>
                                                            <td>{{ $info_data->no_telepon }}</td>
                                                            <td>
                                                                <form action="{{ route('toko_obat.destroy' , $info_data->id)}}" method="POST">
                                                                    @method('DELETE')
                                                                    {{ csrf_field() }}
                                                                    <a href="{{ route('toko_obat.edit', $info_data->id) }}" class="btn btn-sm btn-warning"><i class="fas fa-pencil"></i> Edit</a>
                                                                    <button class="btn btn-sm btn-danger"><i class="fas fa-trash"></i> Delete</button>
                                                                </form>
                                                            </td>
                                                        </tr>
                                                        @endforeach
                                                        @endif
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
</div>
@endsection

@section('fscripts')

@endsection
