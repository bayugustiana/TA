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
                        <h5 class="m-b-10">History</h5>
                        <p class="m-b-0">Data History Peternakan</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">History</a>
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
                        <div class="col-lg-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Data Seluruh Rekam Medis</h5>
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
                                            <a href="#" class="btn btn-sm btn-primary"><i class="fas fa-plus"></i> Tambah</a>
                                        </div>
                                    </div>
                                    <div class="row">
                                        <div class="col-xl-12">
                                            <div class="dt-responsive table-responsive">
                                                <table id="simpletable"
                                                    class="table table-striped table-bordered nowrap">
                                                    <thead>
                                                        <tr>
                                                            <th>No</th>
                                                            <th>Nama Pesan</th>
                                                            <th>Sapi</th>
                                                            <th>Obat</th>
                                                            <th>Dokter</th>
                                                            <th>Aksi</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>Sakit Perut</td>
                                                            <td>3 (Tempat B)</td>
                                                            <td>Neuragad (Kapsul)</td>
                                                            <td>Dr. Setiabudi</td>
                                                            <td>
                                                                <a href="#" class="btn btn-sm btn-warning"><i class="fas fa-pencil"></i> Edit</a>
                                                                <a href="#" class="btn btn-sm btn-danger"><i class="fas fa-trash"></i> Delete</a>
                                                            </td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>Sakit Lambung</td>
                                                            <td>2 (Tempat B)</td>
                                                            <td>Protex (Cair)</td>
                                                            <td>Dr. Setiabudi</td>
                                                            <td>
                                                                <a href="#" class="btn btn-sm btn-warning"><i class="fas fa-pencil"></i> Edit</a>
                                                                <a href="#" class="btn btn-sm btn-danger"><i class="fas fa-trash"></i> Delete</a>
                                                            </td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-lg-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Data Seluruh Transaksi Obat</h5>
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
                                                        <tr>
                                                            <th>ID</th>
                                                            <th>Tanggal Transaksi</th>
                                                            <th>ID Peternakan</th>
                                                            <th>ID Obat</th>
                                                            <th>Total Harga</th>
                                                            <th>Paket</th>
                                                            <th>Status</th>
                                                            <th>Aksi</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>1 Maret 2020</td>
                                                            <td>1 (PT. Aio)</td>
                                                            <td>1 (Neuragad)</td>
                                                            <td>170000</td>
                                                            <td>Box</td>
                                                            <td>Lunas</td>
                                                            <td>
                                                                <a href="#" class="btn btn-sm btn-warning"><i class="fas fa-pencil"></i> Edit</a>
                                                                <a href="#" class="btn btn-sm btn-danger"><i class="fas fa-trash"></i> Delete</a>
                                                            </td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>1 Maret 2020</td>
                                                            <td>1 (PT. Aio)</td>
                                                            <td>1 (Neuragad)</td>
                                                            <td>130000</td>
                                                            <td>Box</td>
                                                            <td>Lunas</td>
                                                            <td>
                                                                <a href="#" class="btn btn-sm btn-warning"><i class="fas fa-pencil"></i> Edit</a>
                                                                <a href="#" class="btn btn-sm btn-danger"><i class="fas fa-trash"></i> Delete</a>
                                                            </td>
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
                    <div class="row">
                        <div class="col-xl-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Data Aktivitas Sapi</h5>
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
                                                        <tr>
                                                            <th>No</th>
                                                            <th>Sapi</th>
                                                            <th>Latitude</th>
                                                            <th>Longitude</th>
                                                            <th>Suhu (Celsius)</th>
                                                            <th>Detak Jantung (Rate)</th>
                                                            <th>Kondisi (Rate)</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        @if(@$info_kesehatan)
                                                        @foreach ($info_kesehatan as $info_data)
                                                        <tr>
                                                            <td>{{ $info_data['id'] }}</td>
                                                            <td>Sapi {{ $info_data['id_sapi'] }}</td>
                                                            <td>{{ $info_data['latitude'] }}</td>
                                                            <td>{{ $info_data['longitude'] }}</td>
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

        L.marker([-6.938487, 107.622548]).addTo(mymap)
            .bindPopup("My Location").openPopup();

        L.circle([-6.941491, 107.629087], 100, {
            color: 'red',
            fillColor: '#f03',
            fillOpacity: 0.5
        }).addTo(mymap).bindPopup("My School");

        L.polygon([
            [-6.938326, 107.626126],
            [-6.937559, 107.627349],
            [-6.939028, 107.627627]
        ]).addTo(mymap).bindPopup("My Friends House");


        var popup = L.popup();

        function onMapClick(e) {
            popup
                .setLatLng(e.latlng)
                .setContent("You clicked the map at " + e.latlng.toString())
                .openOn(mymap);
        }

        mymap.on('click', onMapClick);

    }, 350);
</script>
@endsection
