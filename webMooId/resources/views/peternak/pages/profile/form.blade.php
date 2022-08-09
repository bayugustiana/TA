@extends('peternak.layouts.app')

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
                        <h5 class="m-b-10">Ubah Profile</h5>
                        <p class="m-b-0">Form Profile</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Profile</a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Ubah Profile</a>
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
                                    <h5>Profile</h5>
                                </div>
                                <div class="card-block">
                                    <form role="form" class="form-material" action="{{ route('profile.update') }}" enctype="multipart/form-data" method="POST">
                                        @if (@$info)
                                            @method('PATCH')
                                        @endif
                                        @csrf
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="form-group">
                                                    <div class="row">
                                                        <img class="col-sm-3 user-img img-radius"
                                                        src="{{ (@$info->avatar) ? storage_path('public/peternakan/'.$info->id.'/' . $info->avatar) : asset('assets/images/avatar-0.png') }}" alt="user-img">
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
                                                <div class="form-group form-default">
                                                    <input type="text" name="nama_peternakan" class="form-control fill" value="{{ @$info->nama_peternakan }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Nama Peternakan</label>
                                                    @if ($errors->has('nama_peternakan'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('nama_peternakan') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <div class="row">
                                                        <label class="col-sm-3 col-form-label">Jenis Peternakan</label>
                                                        <div class="col-sm-9">
                                                            <select name="jenis_peternakan" class="form-control">
                                                                <option value="">--SILAHKAN DIPILIH--</option>
                                                                <option value="perorang" {{ (@$info->jenis_peternakan=='perorang') ? 'selected' : '' }}>Perorang</option>
                                                                <option value="kelompok" {{ (@$info->jenis_peternakan=='kelompok') ? 'selected' : '' }}>Kelompok</option>
                                                                <option value="perusahaan" {{ (@$info->jenis_peternakan=='perusahaan') ? 'selected' : '' }}>Perusahaan</option>
                                                            </select>
                                                        </div>
                                                    </div>
                                                    @if ($errors->has('nama_peternakan'))
                                                    <div class="row">
                                                        <div class="col-sm-12">
                                                            <div class="col-form-label text-danger">{{ $errors->first('jenis_peternakan') }}</div>
                                                        </div>
                                                    </div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <textarea class="form-control fill" name="alamat_lengkap">{{ @$info->alamat_lengkap }}</textarea>
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Alamat</label>
                                                    @if ($errors->has('alamat_lengkap'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('alamat_lengkap') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="text" name="no_telepon" class="form-control fill" value="{{ @$info->no_telepon }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Nomor Telepon</label>
                                                    @if ($errors->has('no_telepon'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('no_telepon') }}</div>
                                                    @endif
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <div class="form-group form-default">
                                                    <input type="text" name="nama_lengkap" class="form-control fill" value="{{ @$info->nama_lengkap }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Nama Lengkap Pemilik</label>
                                                    @if ($errors->has('nama_lengkap'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('nama_lengkap') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <input type="text" name="username" class="form-control fill" value="{{ @$info->username }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Username</label>
                                                    @if ($errors->has('username'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('username') }}</div>
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
                                                <div class="form-group form-default">
                                                    <input type="text" name="email" class="form-control fill" value="{{ @$info->email }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Email</label>
                                                    @if ($errors->has('email'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('email') }}</div>
                                                    @endif
                                                </div>
                                                <div class="form-group form-default">
                                                    <input id="areaput" type="text" name="area" class="form-control fill" value="{{ @$info->area }}">
                                                    <span class="form-bar"></span>
                                                    <label class="float-label">Area</label>
                                                    @if ($errors->has('area'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('area') }}</div>
                                                    @endif
                                                    @if ($errors->has('latitude'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('latitude') }}</div>
                                                    @endif
                                                    @if ($errors->has('longitude'))
                                                    <div class="col-form-label text-danger">{{ $errors->first('longitude') }}</div>
                                                    @endif
                                                    <input id="latput" type="text" name="latitude" class="form-control" value="{{ @$info->latitude }}" readonly hidden>
                                                    <input id="lonput" type="text" name="longitude" class="form-control" value="{{ @$info->longitude }}" readonly hidden>
                                                </div>
                                                <h4 class="text-center">Tentukan lokasi dengan map</h4>
                                            </div>
                                        </div>
                                        <div class="row mb-2">
                                            <div class="col-md-12">
                                                <div id="map"></div>
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
<script src="https://unpkg.com/leaflet@1.6.0/dist/leaflet.js"
  integrity="sha512-gZwIG9x3wUXg2hdXF6+rVkLF/0Vi9U8D2Ntg4Ga5I5BZpVkVxlJWbSQtXPSiUTtC0TjtGOmxa1AJPuV0CPthew=="
  crossorigin=""></script>

  <script>
    setTimeout(function () {
        // Leaflet
        var latitude = {!! @$info->latitude !!};
        var longitude = {!! @$info->longitude !!};
        var area = {!! @$info->area !!} // 1 meter = 1 unit

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

        //Buat Layer
        var ternakLayer = L.layerGroup();
        mymap.addLayer(ternakLayer);

        //Startup Layer
        L.marker([latitude, longitude]).addTo(ternakLayer)
            .bindPopup("Peternakan Saya").openPopup();
        L.circle([latitude, longitude], area, {
            color: 'transparent',
            fillColor: 'green',
            fillOpacity: 0.5
        }).addTo(ternakLayer).bindPopup("Daerah Ternak Saya");
        var popup = L.popup();

        // OnClick Refresh
        function onMapClick(e) {
            var eventLoc = e.latlng.toString();
            var message = "You clicked the map at ";
            popup
                .setLatLng(e.latlng)
                .setContent(message + eventLoc)
                .openOn(ternakLayer);
            var end = eventLoc.length;
            var result = eventLoc.substr(7,end).split(",");
            $('#latput').val(result[0]);
            $('#lonput').val(result[1].slice(1,-2));

            // Refresh Clear
            ternakLayer.clearLayers();

            // Set Marker + Area
            var area_input = Number($('#areaput').val()); // 1 meter = 1 unit

            L.marker([result[0], result[1].slice(1,-2)]).addTo(ternakLayer)
            .bindPopup("Peternakan Saya").openPopup();
            L.circle([result[0], result[1].slice(1,-2)], area_input, {
                color: 'transparent',
                fillColor: 'green',
                fillOpacity: 0.5
            }).addTo(ternakLayer).bindPopup("Daerah Ternak Baru Saya");
        }
        mymap.on('click', onMapClick);
    }, 350);
</script>
@endsection
