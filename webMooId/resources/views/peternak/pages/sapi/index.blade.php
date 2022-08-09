@extends('peternak.layouts.app', ['activeMenu' => 'sapi'])

@section('hstyles')
<link rel="stylesheet" type="text/css"
    href="{{ asset('bower_components/datatables.net-bs4/css/dataTables.bootstrap4.min.css') }}">

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
                        <h5 class="m-b-10">Sapi</h5>
                        <p class="m-b-0">Data Sapi Peternakan</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Sapi</a></li>
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
                    @if(@$info_kesehatan)
                    <div class="row">
                        <div class="col-xl-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5><a href="{{ route('sapi.aktivitas') }}">Grafik Kesehatan Sapi Pekan Ini</a></h5>
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
                                            <div id="sapi-barchart" style="height: 245px"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        @if(@$info_rekammedis)
                        <div class="col-xl-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Grafik Rekam Medis Pekan ini</h5>
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
                                            <div id="sapisakit-barchart" style="height: 245px"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        @endif
                    </div>
                    @endif
                    <div class="row">
                        <div class="col-xl-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Data Seluruh Sapi</h5>
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
                                            <a href="{{ route('sapi.create') }}" class="btn btn-sm btn-primary"><i class="fas fa-plus"></i> Tambah</a>
                                        </div>
                                    </div>
                                    @if(@$info_kesehatan)
                                    <div class="row mb-3">
                                        <div class="col-xl-12">
                                            <div id="map"></div>
                                        </div>
                                    </div>
                                    @endif
                                    <div class="row">
                                        <div class="col-xl-12">
                                            <div class="dt-responsive table-responsive">
                                                <table id="simpletable"
                                                    class="table table-striped table-bordered nowrap">
                                                    <thead>
                                                        <tr>
                                                            <th>ID Sapi</th>
                                                            <th>Jenis Tempat</th>
                                                            <th>Tanggal Lahir</th>
                                                            <th>Jenis Sapi</th>
                                                            <th>Tipe Sapi</th>
                                                            <th>Jenis Kelamin</th>
                                                            @if(@$info_kesehatan)
                                                            <th>Lokasi (Lat,Lon)</th>
                                                            @endif
                                                            <th>Aksi</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        @php
                                                        $i = 1;
                                                        @endphp
                                                        @if(@$info_sapi)
                                                        @foreach($info_sapi as $info_data)
                                                        <tr>
                                                            <td>Sapi {{ $info_data->id }}</td>
                                                            <td>{{ $info_data->sapi->tempat->jenis_tempat }}</td>
                                                            <td>{{ $info_data->sapi->tanggal_lahir }}</td>
                                                            <td>{{ $info_data->sapi->jenis_sapi }}</td>
                                                            <td>{{ $info_data->sapi->tipe_sapi }}</td>
                                                            <td>{{ $info_data->sapi->jenis_kelamin }}</td>
                                                            @if(@$info_data->latitude && @$info_data->longitude)
                                                            @php
                                                            if ($i == 1) {
                                                                @$first_lat = @$info_data->latitude;
                                                                @$first_lon = @$info_data->longitude;
                                                                $i++;
                                                            }
                                                            @endphp
                                                            <td>{{ @$info_data->latitude }}, {{ @$info_data->longitude }}</td>
                                                            @else
                                                            <td>--Belum Ada Data--</td>
                                                            @endif
                                                            <td>
                                                                <form action="{{ route('sapi.destroy' , $info_data->sapi->id)}}" method="POST">
                                                                    @method('DELETE')
                                                                    {{ csrf_field() }}
                                                                    <a href="{{ route('sapi.edit', $info_data->sapi->id) }}" class="btn btn-sm btn-warning"><i class="fas fa-pencil"></i> Edit</a>
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
<script>
    $(document).ready(function () {
    var sapi_barchart = AmCharts.makeChart("sapi-barchart", {
        "type": "serial",
        "theme": "light",
        "dataProvider": {!! $info_kesehatan !!},
        "valueAxes": [{
            "gridAlpha": 0.3,
            "gridColor": "yellow",
            "axisColor": "transparent",
            "color": 'green',
            "dashLength": 0
        }],
        "gridAboveGraphs": true,
        "startDuration": 1,
        "graphs": [{
            "balloonText": "Jumlah Sapi: <b>[[value]]</b>",
            "fillAlphas": 1,
            "lineAlpha": 1,
            "lineColor": "#36d3d6",
            "type": "column",
            "valueField": "count",
            "columnWidth": 0.5
        }],
        "chartCursor": {
            "categoryBalloonEnabled": false,
            "cursorAlpha": 0,
            "zoomable": false
        },
        "categoryField": "type",
        "categoryAxis": {
            "gridPosition": "start",
            "gridAlpha": 0,
            "axesAlpha": 0,
            "lineAlpha": 0,
            "fontSize": 12,
            "color": '#000',
            "tickLength": 0
        },
        "export": {
            "enabled": false
        }
    });
});
</script>
<script src="https://unpkg.com/leaflet@1.6.0/dist/leaflet.js"
  integrity="sha512-gZwIG9x3wUXg2hdXF6+rVkLF/0Vi9U8D2Ntg4Ga5I5BZpVkVxlJWbSQtXPSiUTtC0TjtGOmxa1AJPuV0CPthew=="
  crossorigin=""></script>

  <script>
    setTimeout(function () {
        // Leaflet
        var mymap = L.map('map').setView([{!! @$first_lat !!}, {!! @$first_lon !!}], 13);
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
        @foreach($info_sapi as $info_data)
        @if(@$info_data->latitude)
        L.marker([{!! @$info_data->latitude !!}, {!! @$info_data->longitude !!}]).addTo(mymap)
            .bindPopup("Sapi "+"{!! @$info_data->sapi->id !!}").openPopup();
        @endif
        @endforeach
        var popup = L.popup();
        function onMapClick(e) {
            var eventLoc = e.latlng.toString();
            var message = "You clicked the map at ";
            popup
                .setLatLng(e.latlng)
                .setContent(message + eventLoc)
                .openOn(mymap);
        }
        mymap.on('click', onMapClick);
    }, 350);
</script>
<script type="text/javascript" src="{{ asset('assets/pages/dashboard/my-dashboard.js') }}"></script>
<script src="{{ asset('assets/js/jquery.mCustomScrollbar.concat.min.js') }}"></script>

<script src="{{ asset('assets/pages/chart/float/jquery.flot.js') }}"></script>
<script src="{{ asset('assets/pages/chart/float/jquery.flot.categories.js') }}"></script>
<script src="{{ asset('assets/pages/chart/float/curvedLines.js') }}"></script>
<script src="{{ asset('assets/pages/chart/float/jquery.flot.tooltip.min.js') }}"></script>

<script src="{{ asset('assets/pages/widget/amchart/amcharts.js') }}"></script>
<script src="{{ asset('assets/pages/widget/amchart/gauge.js') }}"></script>
<script src="{{ asset('assets/pages/widget/amchart/serial.js') }}"></script>
<script src="{{ asset('assets/pages/widget/amchart/light.js') }}"></script>
<script src="{{ asset('assets/pages/widget/amchart/pie.min.js') }}"></script>

<script type="text/javascript" src="{{ asset('assets/pages/widget/excanvas.js') }}"></script>

<script src="{{ asset('assets/pages/waves/js/waves.min.js') }}"></script>
@endsection
