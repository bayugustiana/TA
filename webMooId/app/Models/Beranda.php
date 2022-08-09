<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Carbon;
use App\Models\Peternakan;

class Beranda extends Model
{
    public function total_sapi()
    {
        return Peternakan::find(1)->sapi->count();
    }
    public function total_tempat()
    {
        return Peternakan::find(1)->tempat->count();
    }
    public function total_petugas()
    {
        return Peternakan::find(1)->petugas->count();
    }
    public function total_dokter()
    {
        return Peternakan::find(1)->dokter->count();
    }
    public function total_artikel()
    {
        return DB::table('artikel')->count();
    }
    public function total_transaksi()
    {
        return Peternakan::find(1)->transaksi->count();
    }
    public function total_toko_obat()
    {
        return DB::table('toko_obat')->count();
    }
    public function status_ternak()
    {
        return DB::table('toko_obat')->count();
    }
    public function tampil_mingguan()
    {
        $data = DB::table('rekam_medis')->join('sapi', 'sapi.id', '=', 'rekam_medis.id_sapi')->join('tempat', 'tempat.id', '=', 'sapi.id_tempat')->join('peternakan', 'peternakan.id', '=', 'tempat.id_peternakan')->select('id_sapi', 'id_tempat', 'jenis_tempat', DB::raw('count(*) as total'))->groupBy('id_sapi')->where('tempat.id_peternakan', '1')->whereBetween('waktu', [Carbon::now()->startOfWeek(), Carbon::now()->endOfWeek()])->get();
        return $data;
    }
}
