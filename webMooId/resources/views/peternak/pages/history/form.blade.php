@extends('peternak.layouts.app', ['activeMenu' => 'history'])

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
                        <h5 class="m-b-10">{{ @$post ? 'Ubah' : 'Tambah' }} Rekam Medis</h5>
                        <p class="m-b-0">Form Rekam Medis Peternakan</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">History</a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">{{ @$post ? 'Ubah' : 'Tambah' }} Rekam Medis</a>
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
                                    <h5>Rekam Medis</h5>
                                </div>
                                <div class="card-block">
                                    <form class="form-material">
                                        <div class="row">
                                            <div class="col-md-12">
                                                <div class="form-group row">
                                                    <label class="col-sm-2 col-form-label">ID Obat</label>
                                                    <div class="col-sm-10">
                                                        <select name="id_obat" class="form-control">
                                                            <option value="">--SILAHKAN DIPILIH--</option>
                                                            <option value="1">1 - (Neuragad)</option>
                                                            <option value="2">2 - (Betadine)</option>
                                                        </select>
                                                    </div>
                                                </div>
                                                <div class="form-group row">
                                                    <label class="col-sm-2 col-form-label">ID Dokter</label>
                                                    <div class="col-sm-10">
                                                        <select name="id_obat" class="form-control">
                                                            <option value="">--SILAHKAN DIPILIH--</option>
                                                            <option value="1">1 - (Dr. Setiabudi)</option>
                                                            <option value="2">2 - (Dr. Mariawati)</option>
                                                        </select>
                                                    </div>
                                                </div>
                                                <div class="form-group row">
                                                    <label class="col-sm-2 col-form-label">ID Sapi</label>
                                                    <div class="col-sm-10">
                                                        <select name="id_obat" class="form-control">
                                                            <option value="">--SILAHKAN DIPILIH--</option>
                                                            <option value="1">1 - (Tempat B)</option>
                                                            <option value="2">2 - (Tempat C)</option>
                                                        </select>
                                                    </div>
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="text" name="nama_pesan" class="form-control">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Nama Pesan</label>
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

@endsection
