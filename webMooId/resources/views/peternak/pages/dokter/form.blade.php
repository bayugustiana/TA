@extends('peternak.layouts.app', ['activeMenu' => 'dokter'])

@section('hstyles')
<link rel="stylesheet" href="{{ asset('bower_components/tempusdominus-bootstrap-4/css/tempusdominus-bootstrap-4.min.css') }}" />
@endsection

@section('content')
<div class="pcoded-content">
    <!-- PAGE HEADER -->
    <div class="page-header">
        <div class="page-block">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <div class="page-header-title">
                        <h5 class="m-b-10">{{ @$info ? 'Ubah' : 'Tambah' }} Dokter</h5>
                        <p class="m-b-0">Form Dokter</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Dokter</a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">{{ @$info ? 'Ubah' : 'Tambah' }} Dokter</a>
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
                                    <h5>Dokter</h5>
                                </div>
                                <div class="card-block">
                                    <form class="form-material">
                                        @if (@$info)
                                            @method('PATCH')
                                        @endif
                                        @csrf
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="form-group form-default">
                                                    <input type="text" name="nama"
                                                    class="form-control {{ (@$info->nama) ? 'fill' : '' }}" value="{{ @$info->nama }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Nama Lengkap</label>
                                                    @if ($errors->has('nama'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('nama') }}</div>
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
                                                <div class="form-group form-default">
                                                    <input type="password" name="footer-email" class="form-control">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Password</label>
                                                    @if ($errors->has('password'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('password') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="password" name="confirm_password" class="form-control">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Konfirmasi Password</label>
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <div class="form-group form-default">
                                                    <input type="text" name="no_telepon" class="form-control {{ (@$info->no_telepon) ? 'fill' : '' }}" value="{{ @$info->no_telepon }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Nomor Telepon</label>
                                                    @if ($errors->has('no_telepon'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('no_telepon') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="text" name="pendidikan_terakhir" class="form-control {{ (@$info->pendidikan_terakhir) ? 'fill' : '' }}" value="{{ @$info->pendidikan_terakhir }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Pendidikan Terakhir</label>
                                                    @if ($errors->has('pendidikan_terakhir'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('pendidikan_terakhir') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="text" name="tempat_kerja" class="form-control {{ (@$info->tempat_kerja) ? 'fill' : '' }}" value="{{ @$info->tempat_kerja }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Tempat Kerja</label>
                                                    @if ($errors->has('tempat_kerja'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('tempat_kerja') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="number" name="pengalaman_kerja" class="form-control {{ (@$info->pengalaman_kerja) ? 'fill' : '' }}" value="{{ @$info->pengalaman_kerja }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Pengalaman Kerja (1 bln = 1)</label>
                                                    @if ($errors->has('pengalaman_kerja'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('pengalaman_kerja') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group">
                                                    <div class="row">
                                                        <img class="col-sm-3 user-img img-radius"
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
                                                        <div class="col-sm-12">
                                                            @if ($errors->has('avatar'))
                                                            <div class="col-form-label text-danger">{{ $errors->first('avatar') }}</div>
                                                            @endif
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="row">
                                            <div class="col-12">
                                                <div class="card">
                                                    <div class="card-header">
                                                        <h5>Deskripsi Pribadi</h5>
                                                    </div>
                                                    <div class="card-block">
                                                        <div class="form-group form-default">
                                                            <textarea class="form-control {{ (@$info->tentang) ? 'fill' : '' }}" name="tentang">{{ @$info->tentang }}</textarea>
                                                            <span class="form-bar"></span>
                                                            <label class="float-label">Tentang</label>
                                                            @if ($errors->has('tentang'))
                                                            <div class="col-form-label text-danger">{{ $errors->first('tentang') }}</div>
                                                            @endif
                                                        </div>
                                                    </div>
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
<script type="text/javascript">
    $(function () {
        $('#tanggallahir').datetimepicker({
            format : 'YYYY-MM-DD',
            ignoreReadonly: true
        });
    });
</script>
<script type="text/javascript" src="{{ asset('bower_components/moment/js/moment.min.js') }}"></script>
<script type="text/javascript" src="{{ asset('bower_components/tempusdominus-bootstrap-4/js/tempusdominus-bootstrap-4.min.js') }}"></script>
@endsection
