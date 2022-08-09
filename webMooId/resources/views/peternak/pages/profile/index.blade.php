@extends('peternak.layouts.app')

@section('hstyles')
<link rel="stylesheet" type="text/css"
    href="{{ asset('assets/pages/advance-elements/css/bootstrap-datetimepicker.css') }}">

<link rel="stylesheet" type="text/css"
    href="{{ asset('bower_components/bootstrap-daterangepicker/css/daterangepicker.css') }}" />

<link rel="stylesheet" type="text/css" href="{{ asset('bower_components/datedropper/css/datedropper.min.css') }}" />

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
                        <h5 class="m-b-10">Profile</h5>
                        <p class="m-b-0">My Profile</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Profile</a></li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    <div class="pcoded-inner-content">
        <div class="main-body">
            <div class="page-wrapper">

                <div class="page-body">

                    <div class="row">
                        <div class="col-lg-12">
                            <div class="cover-profile">
                                <div class="profile-bg-img">
                                    <img class="profile-bg-img img-fluid"
                                        src="{{ asset('assets/images/user-profile/bg-img1.jpg') }}" alt="bg-img">
                                    <div class="card-block user-info">
                                        <div class="col-md-12">
                                            <div class="media-left">
                                                <a href="#" class="profile-image">
                                                    <img class="user-img img-radius"
                                                        src="{{ asset('assets/images/avatar-0.png') }}" alt="user-img"
                                                        width="100px">
                                                </a>
                                            </div>
                                            <div class="media-body row">
                                                <div class="col-lg-12">
                                                    <div class="user-title">
                                                        <h2>{{ @$info->nama_lengkap }}</h2>
                                                        <span class="text-white">Owner Of {{ @$info->nama_peternakan }}</span>
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
                                    <h5 class="card-header-text">About Me</h5>
                                    <a href="{{ route('profile.edit') }}" class="btn btn-sm btn-primary waves-effect waves-light f-right">
                                        <i class="fas fa-user-edit"></i>
                                    </a>
                                </div>
                                <div class="card-block">
                                    <div class="view-info">
                                        <div class="row">
                                            <div class="col-lg-12">
                                                <div class="general-info">
                                                    <div class="row">
                                                        <div class="col-lg-12 col-xl-6">
                                                            <div class="table-responsive">
                                                                <table class="table m-0">
                                                                    <tbody>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Nama Peternakan
                                                                            </th>
                                                                            <td>{{ @$info->nama_peternakan }}</td>
                                                                        </tr>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Jenis Peternakan</th>
                                                                            <td>{{ @$info->jenis_peternakan }}</td>
                                                                        </tr>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Alamat Peternakan
                                                                            </th>
                                                                            <td>{{ @$info->alamat_lengkap }}</td>
                                                                        </tr>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Nomor Telepon</th>
                                                                            <td>{{ @$info->no_telepon }}</td>
                                                                        </tr>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Email
                                                                            </th>
                                                                            <td>{{ @$info->email }}</td>
                                                                        </tr>
                                                                    </tbody>
                                                                </table>
                                                            </div>
                                                        </div>

                                                        <div class="col-lg-12 col-xl-6">
                                                            <div class="table-responsive">
                                                                <table class="table">
                                                                    <tbody>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Nama Lengkap Pemilik
                                                                            </th>
                                                                            <td>{{ @$info->nama_lengkap }}
                                                                            </td>
                                                                        </tr>
                                                                        <tr>
                                                                            <th scope="row">
                                                                                Username</th>
                                                                            <td>{{ @$info->username }}</td>
                                                                        </tr>
                                                                    </tbody>
                                                                </table>
                                                            </div>
                                                            <div class="row mb-2">
                                                                <div class="col-12">
                                                                    <h5>Lokasi</h5>
                                                                </div>
                                                            </div>
                                                            <div class="row">
                                                                <div class="col-12">
                                                                    <div id="map"></div>
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

                </div>
            </div>
        </div>

    </div>
</div>
</div>
@endsection

@section('fscripts')
<script src="{{ asset('bower_components/datatables.net/js/jquery.dataTables.min.js') }}"></script>
<script src="{{ asset('bower_components/datatables.net-bs4/js/dataTables.bootstrap4.min.js') }}"></script>
<script src="{{ asset('bower_components/datatables.net-responsive/js/dataTables.responsive.min.js') }}"></script>
<script src="{{ asset('bower_components/datatables.net-responsive-bs4/js/responsive.bootstrap4.min.js') }}"></script>

<script src="{{ asset('assets/pages/ckeditor/ckeditor.js') }}"></script>

<script src="{{ asset('assets/pages/chart/echarts/js/echarts-all.js') }}" type="text/javascript"></script>
<script src="{{ asset('assets/pages/user-profile.js') }}"></script>

<script src="https://unpkg.com/leaflet@1.6.0/dist/leaflet.js"
  integrity="sha512-gZwIG9x3wUXg2hdXF6+rVkLF/0Vi9U8D2Ntg4Ga5I5BZpVkVxlJWbSQtXPSiUTtC0TjtGOmxa1AJPuV0CPthew=="
  crossorigin=""></script>

  <script>
    setTimeout(function () {
        // Leaflet
        var latitude = {!! @$info->latitude !!};
        var longitude = {!! @$info->longitude !!};
        var area = {!! @$info->area !!};
        var mymap = L.map('map').setView([latitude, longitude], 17);
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
        L.marker([latitude, longitude]).addTo(mymap)
            .bindPopup("Peternakan Saya").openPopup();
        L.circle([latitude, longitude], area, {
            color: 'red',
            fillColor: '#f03',
            fillOpacity: 0.5
        }).addTo(mymap).bindPopup("Daerah Ternak Saya");
    }, 350);
</script>
@endsection
