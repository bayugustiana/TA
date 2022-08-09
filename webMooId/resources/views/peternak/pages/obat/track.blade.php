@extends('peternak.layouts.app', ['activeMenu' => 'obat'])

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
                        <h5 class="m-b-10">Lacak Pesanan Saya</h5>
                        <p class="m-b-0">Data Lacak Pesanan Konfirmasi</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Obat</a></li>
                        <li class="breadcrumb-item"><a href="#">Lacak Pesanan</a></li>
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
                                    <h5>Data Lacak Pesanan</h5>
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
                                                            <th>Gambar</th>
                                                            <th>Obat</th>
                                                            <th>Transaksi Toko</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        @if(@$info_transaksi_diproses)
                                                        @foreach ($info_transaksi_diproses as $info_data)
                                                        <tr>
                                                            <td>
                                                                <img width="50" src="{{ (@$info_data->obat->avatar) ? storage_path('public/obat/'. $info_data->id.'/'.$info_data->avatar) : asset('assets/images/product/p1.jpg') }}"
                                                                class="img-fluid"
                                                                alt="tbl"><br>
                                                                @if(checkIfThreeDayPassed($info_data->tanggal_transaksi))
                                                                <a href="#" class="label label-danger"><i class="fad fa-comment-exclamation"></i> Komplain</a>
                                                                @else
                                                                <a href="#" class="label label-info disabled"><i class="ti-reload rotate-refresh"></i> Diproses</a>
                                                                @endif
                                                            </td>
                                                            <td>{{ $info_data->obat->nama_obat }} - ({{ $info_data->packaging }})<br><span class="font-italic">{{ $info_data->obat->tipe_obat }}</span><br><span class="font-weight-bold">Rp. {{ $info_data->obat->harga_obat }}</span></td>
                                                            <td>{{ timeForHuman($info_data->tanggal_transaksi) }}<br><i class="fas fa-store-alt"></i> <span class="font-weight-bold">{{ $info_data->obat->toko_obat->nama_toko_obat }}</span></td>
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
                        <div class="col-lg-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Data Konfirmasi Pesanan</h5>
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
                                                            <th>Gambar</th>
                                                            <th>Obat</th>
                                                            <th>Transaksi Toko</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <!--
                                                            <a href="#" class="btn btn-sm btn-success"><i class="fad fa-box-check"></i> Selesai</a>
                                                            <a href="#" class="btn btn-sm btn-warning"><i class="fad fa-comment-exclamation"></i> Komplain</a>
                                                            <a href="#" class="btn btn-sm btn-info disabled"><i class="fad fa-comment-exclamation"></i> Diproses</a>
                                                        -->
                                                        @if(@$info_transaksi_dikonfirmasi)
                                                        @foreach ($info_transaksi_dikonfirmasi as $info_data)
                                                        <tr>
                                                            <td>
                                                                <img width="50" src="{{ (@$info_data->obat->avatar) ? storage_path('public/obat/'. $info_data->id.'/'.$info_data->avatar) : asset('assets/images/product/p1.jpg') }}"
                                                                class="img-fluid"
                                                                alt="tbl"><br>
                                                                <a href="#" class="label label-success"><i class="fad fa-box-check"></i> Selesai</a>
                                                            </td>
                                                            <td>{{ $info_data->obat->nama_obat }} - ({{ $info_data->packaging }})<br><span class="font-italic">{{ $info_data->obat->tipe_obat }}</span><br><span class="font-weight-bold">Rp. {{ $info_data->obat->harga_obat }}</span></td>
                                                            <td>{{ timeForHuman($info_data->tanggal_transaksi) }}<br><i class="fas fa-store-alt"></i> <span class="font-weight-bold">{{ $info_data->obat->toko_obat->nama_toko_obat }}</span></td>
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

@endsection
