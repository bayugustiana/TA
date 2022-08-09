@extends('peternak.layouts.app', ['activeMenu' => 'obat'])

@section('hstyles')
<link rel="stylesheet" type="text/css" href="{{ asset('bower_components/jquery.steps/css/jquery.steps.css') }}">
@endsection

@section('content')
<div class="pcoded-content">
    <!-- PAGE HEADER -->
    <div class="page-header">
        <div class="page-block">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <div class="page-header-title">
                        <h5 class="m-b-10">Keranjang Pembelian Obat</h5>
                        <p class="m-b-0">Data Keranjang Pembelian Obat</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Obat</a></li>
                        <li class="breadcrumb-item"><a href="#">Keranjang Obat</a></li>
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
                        <div class="col-sm-12">

                            <div class="card">
                                <div class="card-header">
                                    <h5>Shopping Cart</h5>
                                </div>
                                <div class="card-block">
                                    <div class="row">
                                        <div class="col-md-12">
                                            <div id="wizard">
                                                <section>
                                                    <form class="wizard-form" id="basic-forms"
                                                        action="#">

                                                        <h3> Keranjang Pesanan </h3>
                                                        <fieldset>
                                                            <table id="e-product-list"
                                                                class="table table-responsive table-striped dt-responsive nowrap dataTable no-footer dtr-inline cart-page"
                                                                role="grid" style="width: 100%;">
                                                                <thead>
                                                                    <tr>
                                                                        <th class="sorting_disabled"
                                                                            rowspan="1" colspan="1"
                                                                            style="width: 125px;">
                                                                            Image</th>
                                                                        <th class="sorting_disabled"
                                                                            rowspan="1" colspan="1"
                                                                            style="width: 1023px;">
                                                                            Nama Obat</th>
                                                                        <th class="sorting_disabled"
                                                                            rowspan="1" colspan="1"
                                                                            style="width: 1023px;">
                                                                            Tipe Obat</th>
                                                                        <th class="sorting_disabled"
                                                                            rowspan="1" colspan="1"
                                                                            style="width: 100px;">
                                                                            Jumlah</th>
                                                                            <th class="sorting_disabled"
                                                                            rowspan="1" colspan="1"
                                                                            style="width: 153px;">
                                                                            Harga</th>
                                                                        <th class="sorting_disabled"
                                                                            rowspan="1" colspan="1"
                                                                            style="width: 100px;">
                                                                            Total Harga</th>
                                                                        <th class="sorting_disabled"
                                                                            rowspan="1" colspan="1"
                                                                            style="width: 134px;text-align:center">
                                                                            Aksi</th>
                                                                    </tr>
                                                                </thead>
                                                                <tbody>
                                                                    @php
                                                                    $total = 0;
                                                                    @endphp
                                                                    @if(@$info_transaksi)
                                                                    @foreach($info_transaksi as $info_data)
                                                                    <tr>
                                                                        <td class="pro-list-img"
                                                                            tabindex="0">
                                                                            <img src="{{ (@$info_data->obat->avatar) ? storage_path('public/obat/'. $info_data->id.'/'.$info_data->avatar) : asset('assets/images/product/p1.jpg') }}"
                                                                                class="img-fluid"
                                                                                alt="tbl">
                                                                        </td>
                                                                        <td class="pro-name">
                                                                            <h6>{{ $info_data->obat->nama_obat }}</h6>
                                                                        </td>
                                                                        <td>{{ $info_data->obat->tipe_obat }}</td>
                                                                        <td>
                                                                            {{ $info_data->packaging }}
                                                                        </td>
                                                                        <td>Rp {{ $info_data->obat->harga_obat }}</td>
                                                                        <td>
                                                                            Rp {{ $info_data->total_harga }}
                                                                        </td>
                                                                        <td
                                                                            class="action-icon text-center">
                                                                            <a href="#!"
                                                                                class="text-muted"
                                                                                data-toggle="tooltip"
                                                                                data-placement="top"
                                                                                title=""
                                                                                data-original-title="Delete"><i
                                                                                    class="icofont icofont-delete-alt"></i></a>
                                                                        </td>
                                                                    </tr>
                                                                    @php
                                                                    $total += $info_data->total_harga;
                                                                    @endphp
                                                                    @endforeach
                                                                    @endif
                                                                    <tr class="bg-info">
                                                                        <td colspan=5 class="text-right font-weight-bold">Total Keseluruhan Harga</td>
                                                                        <td colspan=3 class="text-left font-weight-bold">Rp {{ $total }}</td>
                                                                    </tr>
                                                                </tbody>
                                                            </table>
                                                        </fieldset>

                                                        <h3> Konfirmasi </h3>
                                                        <fieldset>
                                                            <div class="confirmation">
                                                                <div class="text-primary m-b-20">
                                                                    <i class="fas fa-check-circle"></i>
                                                                </div>
                                                                <div
                                                                    class="confirmation-content text-center">
                                                                    <h3>Selamat! Pesanan anda telah diproses.</h3>
                                                                    <div class="row">
                                                                        <div
                                                                            class="col-md-6 offset-md-3">
                                                                            <p>
                                                                                Mohon untuk menunggu konfirmasi dari toko penjual.
                                                                            </p>
                                                                            <a href="{{ route('obat.track') }}" class="btn btn-primary m-y"><i class="fas fa-truck"></i> Lacak Pesanan</a>
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </fieldset>
                                                    </form>
                                                </section>
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
<script src="{{ asset('bower_components/jquery.cookie/js/jquery.cookie.js') }}"></script>
<script src="{{ asset('bower_components/jquery.steps/js/jquery.steps.js') }}"></script>
<script src="{{ asset('bower_components/jquery-validation/js/jquery.validate.js') }}"></script>

<script src="{{ asset('bower_components/underscore.js/1.8.3/underscore-min.js') }}"></script>
<script src="{{ asset('bower_components/moment.js/moment.min.js') }}"></script>
<script type="text/javascript" src="{{ asset('assets/pages/form-validation/validate.js') }}"></script>

<script src="{{ asset('assets/pages/forms-wizard-validation/form-wizard.js') }}"></script>
@endsection
