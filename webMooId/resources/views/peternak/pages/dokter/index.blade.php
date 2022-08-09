@extends('peternak.layouts.app', ['activeMenu' => 'dokter'])

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
                        <h5 class="m-b-10">Dokter</h5>
                        <p class="m-b-0">Data Seluruh Dokter Hewan</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Dokter</a></li>
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
                        @if(@$info_dokter)
                        @foreach($info_dokter as $info_data)
                        <div class="col-lg-6 col-xl-3 col-md-6">
                            <div class="card rounded-card user-card">
                                <div class="card-block">
                                    <div class="img-hover">
                                        <img class="img-fluid img-radius" src="{{ (@$info_data->avatar) ? storage_path('public/dokter/' . $info_data->avatar) : asset('assets/images/avatar-0.png') }}" alt="round-img">
                                        <div class="img-overlay img-radius">
                                            <span>
                                                <form action="{{ route('dokter.destroy' , $info_data->id)}}" method="POST">
                                                    @method('DELETE')
                                                    {{ csrf_field() }}
                                                    <a href="{{ route('dokter.edit', $info_data->id) }}" class="btn btn-sm btn-warning"><i class="fas fa-pencil"></i></a>
                                                    <button class="btn btn-sm btn-danger"><i class="fas fa-trash"></i></button>
                                                </form>
                                            </span>
                                        </div>
                                    </div>
                                    <div class="user-content">
                                        <a href="{{ route('dokter.show', $info_data->id) }}"><h4>{{ $info_data->nama }}</h4></a>
                                        <p class="m-b-0 text-muted">Dokter Hewan</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        @endforeach
                        @endif
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
@endsection

@section('fscripts')

@endsection
