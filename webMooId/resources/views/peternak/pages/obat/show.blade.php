@extends('peternak.layouts.app', ['activeMenu' => 'obat'])

@section('hstyles')
<link rel="stylesheet" type="text/css" href="{{ asset('bower_components/slick-carousel/css/slick.css') }}">
<link rel="stylesheet" type="text/css" href="{{ asset('bower_components/slick-carousel/css/slick-theme.css') }}">
@endsection

@section('content')
<div class="pcoded-content">

    <div class="page-header">
        <div class="page-block">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <div class="page-header-title">
                        <h5 class="m-b-10">Lihat Detail Obat</h5>
                        <p class="m-b-0">Data Detail Obat</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="index-2.html"> <i class="fa fa-home"></i> </a>
                        </li>
                        <li class="breadcrumb-item"><a href="#!">Obat</a>
                        </li>
                        <li class="breadcrumb-item"><a href="#!">Lihat Obat</a>
                        </li>
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
                        <div class="col-md-12">

                            <div class="card product-detail-page">
                                <div class="card-block">
                                    <div class="row">
                                        <div class="col-lg-5 col-xs-12">
                                            <div class="port_details_all_img row">
                                                <div class="col-lg-12 m-b-15">
                                                    <div id="big_banner">
                                                        <div class="port_big_img">
                                                            <img class="img img-fluid"
                                                                src="{{ (@$info_data->avatar) ? storage_path('public/obat/'. $info_data->id.'/'.$info_data->avatar) : asset('assets/images/product/p1.jpg') }}"
                                                                alt="Big_ Details">
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="col-lg-7 col-xs-12 product-detail"
                                            id="product-detail">
                                            <div class="row">
                                                <div class="col-lg-12">
                                                    <span
                                                        class="txt-muted d-inline-block">ID
                                                        Produk: <a href="#!"> {{ @$info->id }} </a>
                                                    </span>
                                                    <span class="f-right"><i class="fas fa-tablets"></i> {{ (@$info->stok_obat->count() > 0) ? @$info->stok_obat->count().' Buah Tersedia' : 'Habis' }}</span>
                                                </div>
                                                <div class="col-lg-12">
                                                    <h4 class="pro-desc">{{ @$info->nama_obat }}</h4>
                                                </div>
                                                <div class="col-lg-12 mb-1">
                                                    <a href="{{ route('obat.show_toko_obat',@$info->toko_obat->id) }}" class="btn btn-sm btn-primary">
                                                        <i class="fas fa-store-alt"></i> {{ @$info->toko_obat->nama_toko_obat }}
                                                    </a>
                                                </div>
                                                <div class="col-lg-12">
                                                    <span class="label label-info"><i class="fas fa-prescription-bottle"></i> {{ @$info->tipe_obat }}
                                                    </span>
                                                </div>
                                                <div class="col-lg-12">
                                                    <span class="text-primary product-price">Rp. {{ @$info->harga_obat }}</span>
                                                    <hr>
                                                    <p>
                                                        {{ @$info->deskripsi }}
                                                    </p>
                                                    <hr>
                                                    <h6 class="f-16 f-w-600 m-t-10 m-b-10">
                                                        Jumlah</h6>
                                                </div>
                                                <div class="col-xl-3 col-sm-12">
                                                    <div class="p-l-0 m-b-25">
                                                        <div class="input-group">
                                                            <div class="row">
                                                                <div class="col-12">

                                                                </div>
                                                            </div>
                                                            <span class="input-group-btn">
                                                                <button type="button"
                                                                    class="btn btn-default btn-number shadow-none btn-sm"
                                                                    disabled="disabled"
                                                                    data-type="minus"
                                                                    data-field="quant[1]">
                                                                    <span
                                                                        class="icofont icofont-minus m-0"></span>
                                                                </button>
                                                            </span>
                                                            <input type="text" name="quant[1]"
                                                                class="form-control input-number text-center"
                                                                value="1">
                                                            <span class="input-group-btn">
                                                                <button type="button"
                                                                    class="btn btn-default btn-number shadow-none btn-sm"
                                                                    data-type="plus"
                                                                    data-field="quant[1]">
                                                                    <span
                                                                        class="icofont icofont-plus m-0"></span>
                                                                </button>
                                                            </span>
                                                            <sub>Min. pembelian 1pcs.</sub>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div class="col-lg-12 col-sm-12 mob-product-btn">
                                                    <button type="button"
                                                        class="btn btn-primary waves-effect waves-light m-r-20">
                                                        <i class="fas fa-cart-arrow-down"></i><span
                                                            class="m-l-10">Tambah ke Keranjang </span>
                                                    </button>
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
        <div class="main-body">
            <div class="page-wrapper">
                <div class="page-body">
                    <!-- HERE IS THE EXAMPLE CONTENT-->

@endsection

@section('fscripts')
<script type="text/javascript" src="{{ asset('bower_components/slick-carousel/js/slick.min.js') }}"></script>
<script type="text/javascript" src="{{ asset('assets/pages/product-detail/product-detail.js') }}"></script>
@endsection
