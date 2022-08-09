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
                        <h5 class="m-b-10">{{ @$info ? 'Ubah' : 'Tambah' }} Toko Obat</h5>
                        <p class="m-b-0">Form Toko Obat</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Toko Obat</a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">{{ @$info ? 'Ubah' : 'Tambah' }} Toko Obat</a>
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
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Toko Obat</h5>
                                </div>
                                <div class="card-block">
                                    <form role="form" action="{{ @$info ? route('toko_obat.update', @$info->id) : route('toko_obat.store') }}" enctype="multipart/form-data" method="POST" class="form-material">
                                        @if (@$info)
                                            @method('PATCH')
                                        @endif
                                        @csrf
                                        <div class="row">
                                            <img class="col-sm-2 user-img img-radius"
                                            src="{{ (@$info->avatar) ? storage_path('public/dokter/' . $info->avatar) : asset('assets/images/avatar-0.png') }}" alt="user-img"
                                            width="100px">
                                            <div class="col-sm-9">
                                                <div class="row">
                                                    <div class="col-sm-12">
                                                        <label class="col-form-label">Avatar</label>
                                                    </div>
                                                </div>
                                                <div class="row">
                                                    <div class="col-sm-12">
                                                        <input type="file" name="avatar" class="form-control">
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="form-group form-default">
                                                    <input type="text" name="nama_toko_obat" class="form-control {{ (@$info->nama_toko_obat) ? 'fill' : '' }}" value="{{ @$info->nama_toko_obat }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Nama Toko Obat</label>
                                                    @if ($errors->has('nama_toko_obat'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('nama_toko_obat') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="text" name="nama_lengkap" class="form-control {{ (@$info->nama_lengkap) ? 'fill' : '' }}" value="{{ @$info->nama_lengkap }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Nama Lengkap</label>
                                                    @if ($errors->has('nama_lengkap'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('nama_lengkap') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="text" name="email" class="form-control {{ (@$info->email) ? 'fill' : '' }}" value="{{ @$info->email }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Email</label>
                                                    @if ($errors->has('email'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('email') }}</div>
                                                    @endif
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <div class="form-group form-default">
                                                    <input type="password" name="password" class="form-control {{ (@$info->password) ? 'fill' : '' }}" value="{{ @$info->password }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Password</label>
                                                    @if ($errors->has('password'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('password') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="password" name="confirm_password" class="form-control">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Confirm Password</label>
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="text" name="no_telepon" class="form-control {{ (@$info->no_telepon) ? 'fill' : '' }}" value="{{ @$info->no_telepon }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Nomor Telepon</label>
                                                    @if ($errors->has('no_telepon'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('no_telepon') }}</div>
                                                    @endif
                                                </div>
                                            </div>
                                        </div>
                                        <div class="row">
                                            <div class="col-12">
                                                <button type="submit" class="btn btn-{{ @$info ? 'warning' : 'primary' }}">Submit</button>
                                            </div>
                                        </div>
                                    </form>
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
