@extends('peternak.layouts.app', ['activeMenu' => 'sapi'])

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
                        <h5 class="m-b-10">{{ @$info ? 'Ubah' : 'Tambah' }} Sapi</h5>
                        <p class="m-b-0">Form Sapi Peternakan</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Sapi</a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">{{ @$info ? 'Ubah' : 'Tambah' }} Sapi</a>
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
                                    <h5>Sapi</h5>
                                </div>
                                <div class="card-block">
                                    <form role="form" action="{{ @$info ? route('sapi.update', @$info->id) : route('sapi.store') }}" enctype="multipart/form-data" method="POST" class="form-material">
                                        @if (@$info)
                                            @method('PATCH')
                                        @endif
                                        @csrf
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="form-group">
                                                    <div class="row">
                                                        <label class="col-sm-2 col-form-label">ID Tempat</label>
                                                        <div class="col-sm-10">
                                                            <select name="id_tempat" class="form-control">
                                                                <option value="">[id] - [jenis tempat]</option>
                                                                @if(@$info_tempat)
                                                                @foreach ($info_tempat as $info_data)
                                                                <option value="{{ $info_data->id }}" {{ ($info_data->id==@$info->id_tempat) ? 'selected' : '' }}>{{ $info_data->id }} - {{ $info_data->jenis_tempat }}</option>
                                                                @endforeach
                                                                @else
                                                                <option value="">-- TEMPAT TIDAK TERSEDIA --</option>
                                                                @endif
                                                            </select>
                                                        </div>
                                                    </div>
                                                    @if ($errors->has('id_tempat'))
                                                    <div class="row">
                                                        <div class="col-sm-12">
                                                            <div class="col-form-label text-danger">{{ $errors->first('id_tempat') }}</div>
                                                        </div>
                                                    </div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-info form-static-label">
                                                    <input id="tanggallahir" type="text" name="tanggal_lahir" class="form-control" value="{{ (@$info->tanggal_lahir) ? @$info->tanggal_lahir : date('Y-m-d') }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Tanggal Lahir</label>
                                                    @if ($errors->has('tanggal_lahir'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('tanggal_lahir') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group">
                                                    <div class="row">
                                                        <label class="col-sm-2 col-float-label">Jenis Kelamin</label>
                                                        <div class="col-sm-10">
                                                            <select name="jenis_kelamin" class="form-control">
                                                                <option value="">-- SILAHKAN DIPILIH --</option>
                                                                <option value="laki-laki" {{ (@$info->jenis_kelamin=="laki-laki") ? 'selected' : '' }}>Laki-laki</option>
                                                                <option value="perempuan" {{ (@$info->jenis_kelamin=="perempuan") ? 'selected' : '' }}>Perempuan</option>
                                                            </select>
                                                        </div>
                                                    </div>
                                                    @if ($errors->has('jenis_kelamin'))
                                                    <div class="row">
                                                        <div class="col-sm-12">
                                                            <div class="col-form-label text-danger">{{ $errors->first('jenis_kelamin') }}</div>
                                                        </div>
                                                    </div>
                                                    @endif
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <div class="form-group form-default form-static-label">
                                                    <input type="text" name="tipe_sapi"
                                                    class="form-control" value="{{ @$info->tipe_sapi }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Tipe Sapi</label>
                                                    @if ($errors->has('tipe_sapi'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('tipe_sapi') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default form-static-label">
                                                    <input type="text" name="jenis_sapi" class="form-control" value="{{ @$info->jenis_sapi }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Jenis Sapi</label>
                                                    @if ($errors->has('jenis_sapi'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('jenis_sapi') }}</div>
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
