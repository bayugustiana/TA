@extends('peternak.layouts.app', ['activeMenu' => 'tempat'])

@section('hstyles')
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.6.0/dist/leaflet.css"
  integrity="sha512-xwE/Az9zrjBIphAcBb3F6JVqxf46+CDLwfLMHloNu6KEQCAWi6HcDUbeOfBIptF7tcCzusKFjFw2yuvEpDL9wQ=="
  crossorigin=""/>

<style>
    #map {
        width: 100%;
        height: 250px;
    }
</style>
@endsection

@section('content')
<div class="pcoded-content">
    <!-- PAGE HEADER -->
    <div class="page-header">
        <div class="page-block">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <div class="page-header-title">
                        <h5 class="m-b-10">{{ @$info ? 'Ubah' : 'Tambah' }} Tempat</h5>
                        <p class="m-b-0">Form Tempat/Kandang Peternakan</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Tempat</a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">{{ @$info ? 'Ubah' : 'Tambah' }} Tempat</a>
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
                                    <h5>Tempat</h5>
                                </div>
                                <div class="card-block">
                                    <form role="form" action="{{ @$info ? route('tempat.update', @$info->id) : route('tempat.store') }}" enctype="multipart/form-data" method="POST" class="form-material">
                                        @if (@$info)
                                            @method('PATCH')
                                        @endif
                                        @csrf
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="form-group">
                                                    <div class="row">
                                                        <label class="col-sm-2 col-form-label">Jenis Tempat</label>
                                                        <div class="col-sm-10">
                                                            <select name="jenis_tempat" class="form-control">
                                                                <option value="">--SILAHKAN DIPILIH--</option>
                                                                <option value="terbuka" {{ (@$info->jenis_tempat=="terbuka") ? 'selected' : '' }}>Terbuka</option>
                                                                <option value="tertutup" {{ (@$info->jenis_tempat=="tertutup") ? 'selected' : '' }}>Tertutup</option>
                                                            </select>
                                                        </div>
                                                    </div>
                                                    @if ($errors->has('jenis_tempat'))
                                                    <div class="row">
                                                        <div class="col-sm-12">
                                                            <div class="col-form-label text-danger">{{ $errors->first('jenis_tempat') }}</div>
                                                        </div>
                                                    </div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="text" name="kapasitas_sapi" class="form-control {{ (@$info->kapasitas_sapi) ? 'fill' : '' }}" value="{{ @$info->kapasitas_sapi }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Kapasitas Sapi</label>
                                                    @if ($errors->has('kapasitas_sapi'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('kapasitas_sapi') }}</div>
                                                    @endif
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <div class="form-group">
                                                    <div class="row">
                                                        <label class="col-sm-2 col-form-label">Status Gateway</label>
                                                        <div class="col-sm-10">
                                                            <select name="status" class="form-control">
                                                                <option value="">--SILAHKAN DIPILIH--</option>
                                                                <option value="tidak aktif" {{ (@$info->gateway->status=="tidak aktif") ? 'selected' : '' }}>Tidak Aktif</option>
                                                                <option value="aktif" {{ (@$info->gateway->status=="aktif") ? 'selected' : '' }}>Aktif</option>
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
<script src="https://unpkg.com/leaflet@1.6.0/dist/leaflet.js"
  integrity="sha512-gZwIG9x3wUXg2hdXF6+rVkLF/0Vi9U8D2Ntg4Ga5I5BZpVkVxlJWbSQtXPSiUTtC0TjtGOmxa1AJPuV0CPthew=="
  crossorigin=""></script>

  <script>
    setTimeout(function () {
        // Leaflet
        var mymap = L.map('map').setView([-6.938487, 107.622548], 13);

        L.tileLayer('https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token={access_token}', {
            attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, ' +
                '<a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, ' +
                'Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
            id: 'mapbox/streets-v11',
            tileSize: 512,
            zoomOffset: -1,
            maxZoom: 18,
            access_token: 'pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw'
        }).addTo(mymap);

        var popup = L.popup();

        function onMapClick(e) {
            var eventLoc = e.latlng.toString();
            popup
                .setLatLng(e.latlng)
                .setContent("You clicked the map at " + eventLoc)
                .openOn(mymap);
            var raw = eventLoc.length;
            var result = eventLoc.substr(7,raw).split(",");
            $('#latput').val(result[0]);
            $('#lonput').val(result[1].slice(1,-2));
        }

        mymap.on('click', onMapClick);

    }, 350);
</script>
@endsection
