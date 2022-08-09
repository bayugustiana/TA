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
                        <h5 class="m-b-10">Tampil Tempat</h5>
                        <p class="m-b-0">Data Detail Tempat/Gateway/Kandang Peternakan</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Tempat</a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Tampil Tempat</a>
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
                        @if(@$info_lokasi->first()->aktivitas)
                        <div class="col-lg-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Tempat Saya</h5>
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
                                    <div id="map"></div>
                                </div>
                            </div>
                        </div>
                        @endif
                        <div class="col-lg-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-header-text">Detail Tempat</h5>
                                    <a href="#" class="btn btn-sm btn-primary waves-effect waves-light f-right">
                                        <i class="fas fa-user-edit"></i>
                                    </a>
                                </div>
                                <div class="card-block">
                                    <div class="view-info">
                                        <div class="row">
                                            <div class="col-lg-12">
                                                <div class="general-info">
                                                    <div class="row">
                                                        <div class="col-lg-12 col-xl-7">
                                                            <div class="table-responsive">
                                                                <table class="table m-0">
                                                                    <tbody>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                ID
                                                                            </th>
                                                                            <td>{{ @$info->id }}</td>
                                                                        </tr>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Jenis Tempat</th>
                                                                            <td>{{ @$info->jenis_tempat }}</td>
                                                                        </tr>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Sapi / Kapasitas Sapi
                                                                            </th>
                                                                            <td>{{ $info->sapi->count() }} / {{ @$info->kapasitas_sapi }}</td>
                                                                        </tr>
                                                                    </tbody>
                                                                </table>
                                                            </div>
                                                        </div>
                                                        @if(@$info->gateway->status == 'tidak aktif')
                                                        <div class="col-lg-12 col-xl-5">
                                                            <img src="data:image/png;base64, {!! base64_encode(QrCode::format('png')->merge(asset('assets/images/logocircle.png'), .2, true)->errorCorrection('H')->size(200)->generate(@$info->gateway->qrcode)) !!} ">
                                                        </div>
                                                        @endif
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
                                    <h5>Data Sapi</h5>
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
                                    <div class="row">
                                        <div class="col-xl-12">
                                            <div class="dt-responsive table-responsive">
                                                <table id="simpletable"
                                                    class="table table-striped table-bordered nowrap">
                                                    <thead>
                                                        <tr class="text-center">
                                                            <th><i class="fad fa-cow"></i></th>
                                                            <th><i class="fas fa-map-marker-smile"></i></th>
                                                            <th><i class="fas fa-temperature-low"></i>C</th>
                                                            <th><i class="fad fa-heartbeat"></i></th>
                                                            <th><i class="fas fa-flag"></i></th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        @if(@$info_sapi)
                                                        @foreach ($info_sapi as $info_data)
                                                        <tr>
                                                            <td>Sapi {{ $info_data['id_sapi'] }}</td>
                                                            <td>{{ $info_data['latitude'] }}, {{ $info_data['longitude'] }}</td>
                                                            <td>{{ $info_data['suhu'] }}</td>
                                                            <td>{{ $info_data['detak_jantung'] }}</td>
                                                            <td>{{ $info_data['kondisi'] }}</td>
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
<script src="https://unpkg.com/leaflet@1.6.0/dist/leaflet.js"
  integrity="sha512-gZwIG9x3wUXg2hdXF6+rVkLF/0Vi9U8D2Ntg4Ga5I5BZpVkVxlJWbSQtXPSiUTtC0TjtGOmxa1AJPuV0CPthew=="
  crossorigin=""></script>
  <script>
    setTimeout(function () {
        // Leaflet
        var latitude = {!! @$info->peternakan->latitude !!};
        var longitude = {!! @$info->peternakan->longitude !!};
        var area = {!! @$info->peternakan->area !!}

        var mymap = L.map('map').setView([latitude, longitude], area);
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

        //Buat Layer
        var ternakLayer = L.layerGroup();
        mymap.addLayer(ternakLayer);

        //Startup Layer
        L.circle([latitude, longitude], area, {
            color: 'transparent',
            fillColor: 'green',
            fillOpacity: 0.5
        }).addTo(ternakLayer).bindPopup("Daerah Peternakan");

        @foreach($info_lokasi as $info_data)
        @if(@$info_data->aktivitas->last()->latitude)
        L.marker([{!! @$info_data->aktivitas->last()->latitude !!}, {!! @$info_data->aktivitas->last()->longitude !!}]).addTo(mymap)
            .bindPopup("Sapi "+"{!! @$info_data->id !!}").openPopup();
        @endif
        @endforeach
    }, 350);
</script>
@endsection
