@extends('peternak.layouts.app', ['activeMenu' => 'petugas'])

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
                        <h5 class="m-b-10">{{ @$post ? 'Ubah' : 'Tambah' }} Petugas</h5>
                        <p class="m-b-0">Form Petugas Peternakan</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Petugas</a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">{{ @$post ? 'Ubah' : 'Tambah' }} Petugas</a>
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
                                    <h5>Petugas</h5>
                                </div>
                                <div class="card-block">
                                    <form class="form-material">
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="form-group form-default">
                                                    <input type="text" name="nama_petugas"
                                                    class="form-control">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Nama Lengkap</label>
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="text" name="username" class="form-control">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Username</label>
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <div class="form-group form-default">
                                                    <input type="password" name="footer-email" class="form-control">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Password</label>
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="password" name="confirm_password" class="form-control">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Konfirmasi Password</label>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="row">
                                            <div class="col-12">
                                                <button type="submit" class="btn btn-{{ @$post ? 'warning' : 'primary' }}">Submit</button>
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
