@extends('peternak.layouts.app', ['activeMenu' => 'obat'])

@section('hstyles')
<link rel="stylesheet" type="text/css"
    href="{{ asset('assets/pages/advance-elements/css/bootstrap-datetimepicker.css') }}">

<link rel="stylesheet" type="text/css"
    href="{{ asset('bower_components/bootstrap-daterangepicker/css/daterangepicker.css') }}" />

<link rel="stylesheet" type="text/css" href="{{ asset('bower_components/datedropper/css/datedropper.min.css') }}" />

<link rel="stylesheet" type="text/css"
    href="{{ asset('bower_components/datatables.net-bs4/css/dataTables.bootstrap4.min.css') }}">
@endsection

@section('content')
<div class="pcoded-content">
    <!-- PAGE HEADER -->
    <div class="page-header">
        <div class="page-block">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <div class="page-header-title">
                        <h5 class="m-b-10">Halaman Toko</h5>
                        <p class="m-b-0">Toko Obat</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Obat</a></li>
                        <li class="breadcrumb-item"><a href="#">Toko Obat</a></li>
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
                                                        src="{{ (@$info->avatar) ? storage_path('public/toko_obat/'.$info->id.'/' . $info->avatar) : asset('assets/images/avatar-0.png') }}" alt="user-img"
                                                        width="100px">
                                                </a>
                                            </div>
                                            <div class="media-body row">
                                                <div class="col-lg-12">
                                                    <div class="user-title">
                                                        <h2><i class="fas fa-store-alt"></i> {{ @$info->nama_toko_obat }}</h2>
                                                        <span class="text-white"><i class="fas fa-user-check"></i> <span class="font-weight-bold">{{ @$info->nama_lengkap }}</span></span>
                                                    </div>
                                                </div>
                                                <div class="pull-right cover-btn">
                                                    <a href="https://wa.me/{{ @$info->no_telepon }}" class="btn btn-success m-b-10">
                                                        <i class="fab fa-whatsapp"></i> Chat Penjual
                                                    </a>
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
                            <h5 class="font-weight-bold">Produk</h5>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-lg-12">
                            <hr>
                        </div>
                    </div>
                    <div class="row">
                        @if(@$info_obat)
                        @foreach($info_obat as $info_data)
                        @if($info_data->stok_obat->count() > 0)
                        <div class="col-xl-3 col-md-4 col-sm-6 col-12">
                            <div class="card prod-view">
                                <div class="prod-item text-center">
                                    <div class="prod-img">
                                        <div class="option-hover">
                                            <button type="button"
                                                class="btn btn-success btn-icon waves-effect waves-light m-r-15 hvr-bounce-in option-icon">
                                                <i class="icofont icofont-cart-alt f-20"></i>
                                            </button>
                                            <button type="button"
                                                class="btn btn-primary btn-icon waves-effect waves-light m-r-15 hvr-bounce-in option-icon">
                                                <i class="icofont icofont-eye-alt f-20"></i>
                                            </button>
                                            <button type="button"
                                                class="btn btn-danger btn-icon waves-effect waves-light hvr-bounce-in option-icon">
                                                <i class="icofont icofont-heart-alt f-20"></i>
                                            </button>
                                        </div>
                                        <a href="{{ route('obat.show', $info_data->id) }}" class="hvr-shrink">
                                            <img src="{{ (@$info_data->avatar) ? storage_path('public/obat/'. $info_data->id.'/'.$info_data->avatar) : asset('assets/images/product/p1.jpg') }}"
                                                class="img-fluid o-hidden">
                                        </a>
                                    </div>
                                    <div class="prod-info">
                                        <a href="{{ route('obat.show', $info_data->id) }}" class="txt-muted">
                                            <h4>{{ $info_data->nama_obat }}</h4>
                                        </a>
                                        <div class="m-b-10">
                                            <label class="label label-success">
                                                {{ $info_data->stok_obat->count() }} <i class="fab fa-gratipay"></i>
                                            </label>
                                            <a class="text-muted f-w-600">
                                                Tipe: {{ $info_data->tipe_obat }}
                                            </a>
                                        </div>
                                        <span class="prod-price">
                                            Rp. {{ $info_data->harga_obat }}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        @endif
                        @endforeach
                        @endif
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
@endsection
